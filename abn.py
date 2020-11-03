'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import numpy as np
import random
import pickle
import cv2
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from torch.optim import lr_scheduler
from os import path

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

# model_names = default_model_names + customized_models_names
model_names = customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_patience', type=float, default=10)
parser.add_argument('--num_classes', type= int, default=6)

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

# Finetuning
parser.add_argument('--fine_tune', action='store_true', default=False)
parser.add_argument('--freeze_layer', type= int, default=0)
parser.add_argument('--model_path',type=str, default='checkpoints/neu/res50_finetuned/checkpoint.pth.tar')

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

# Random Lib Seed
random.seed(args.manualSeed)

# Numpy Seed
np.random.seed(args.manualSeed)

### CuDNN Seed
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# Torch Seed
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_loss = 100 # loss of the best test acc model
best_epoch =0

def main():
    global best_acc, best_loss, best_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.373, 0.373, 0.373],
                                     std=[0.146, 0.146, 0.146])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ## create model
    # using pretrained imagenet to finetune
    if args.fine_tune:
        print("=> using pre-trained model '{}' to finetune to '{}' classes".format(args.arch, args.num_classes))
        #model = models.__dict__[args.arch](pretrained=True)
        model = models.__dict__[args.arch](num_classes = 1000)
        model = torch.nn.DataParallel(model).cuda()
        if not os.path.exists(args.model_path):
            raise ValueError('Loading Pre-trained Model Failed')
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("\t==> Fine tune by freezing {} layers!".format(str(args.freeze_layer)))

        num_classes = args.num_classes
        num_ftrs_in = model.module.fc.in_features
        model.module.att_conv  = nn.Conv2d(num_ftrs_in, num_classes, kernel_size=1, padding=0,
                               bias=False).cuda()
        model.module.bn_att2 = nn.BatchNorm2d(num_classes).cuda()
        model.module.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                               bias=False).cuda()
        model.module.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                               bias=False).cuda()

        model.module.fc = nn.Linear(num_ftrs_in, num_classes).cuda()


    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    # training from scratch or during eval
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes = args.num_classes)

        if not args.evaluate:
             print("=> Training from scratch ...")
        else:
             print("=> Evaluating Mode...")
        model = torch.nn.DataParallel(model).cuda()


    ## Select # of layers to freeze (default: 0)
    child_counter = 0
    for name, child in model.module.named_children():
        if child_counter < args.freeze_layer:
            print(name,"[child ",child_counter,"] was frozen")
            for name, param in child.named_parameters():
                param.requires_grad = False
        else:
            print(name, "[child ", child_counter,"] was not frozen")
        child_counter += 1

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    title = 'ImageNet-' + args.arch
    # Update optimizer with unfreezed params
    params_to_update = model.parameters()

    if args.freeze_layer != 0:
        params_to_update = []
        for name, param in model.module.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print(name)

    optimizer = optim.Adam(params_to_update, lr=args.lr)
    #optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_sch=  lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience= args.lr_patience)

    ## Resume training or for eval mode
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        resume_folder = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(resume_folder, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, args.epochs, use_cuda)

        if not path.exists(path.join(args.checkpoint, 'output')):
            os.mkdir(path.join(args.checkpoint, 'output'))
#        else:
        shutil.rmtree(os.path.join(args.checkpoint, 'output'))
#            os.mkdir(path.join(args.checkpoint, 'output'))

        shutil.copytree("output/", os.path.join(args.checkpoint, 'output'))
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = (test_acc > best_acc) or ((test_acc == best_acc) and (test_loss < best_loss))
        best_acc = max(test_acc, best_acc)

        if is_best:
            best_loss = test_loss
            best_epoch = epoch+1

        print("Best acc: %f , Epoch: %d, Loss: %f" % (best_acc, best_epoch, best_loss))

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        lr_sch.step(test_loss)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()


    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs, _  = model(inputs)
        #att_loss = criterion(att_outputs, targets)
        per_loss = criterion(outputs, targets)
        loss =  per_loss

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 1))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top3: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    softmax = nn.Softmax()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        count = 0
        info_count = 0

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs, attention = model(inputs)
            outputs = softmax(outputs)
            loss = criterion(outputs, targets)
            attention, fe, per = attention

            c_att = attention.data.cpu()
            c_att = c_att.numpy()
            d_inputs = inputs.data.cpu()
            d_inputs = d_inputs.numpy()

            in_b, in_c, in_y, in_x = inputs.shape
            for item_img, item_att in zip(d_inputs, c_att):

                ## img directories
                out_dir = path.join('output')
                if not path.exists(out_dir):
                    os.mkdir(out_dir)

                if not path.exists(path.join(out_dir, 'attention')):
                    os.mkdir(path.join(out_dir, 'attention'))

                if not path.exists(path.join(out_dir, 'raw')):
                    os.mkdir(path.join(out_dir, 'raw'))

                if not path.exists(path.join(out_dir, 'concat')):
                    os.mkdir(path.join(out_dir, 'concat'))

                if not path.exists(path.join(out_dir, 'mask')):
                    os.mkdir(path.join(out_dir, 'mask'))

                v_img = ((item_img.transpose((1,2,0)) + 0.5 +[0.373, 0.373, 0.373]) * [0.146, 0.146, 0.146])* 256
                v_img = v_img[:, :, ::-1]
                resize_att = cv2.resize(item_att[0], (in_x, in_y))
                resize_att *= 255.
                org = v_img

                cv2.imwrite('stock1.png', v_img)
                cv2.imwrite('stock2.png', resize_att)
                v_img = cv2.imread('stock1.png')
                vis_map = cv2.imread('stock2.png')

                # pure attention map
                vis_map = vis_map - np.min(vis_map)
                vis_map = vis_map/ np.max(vis_map)
                vis_map= np.uint8(256 * vis_map)
                jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

                # convert att map to HSV
                heatmap_BGR = cv2.cvtColor(jet_map, cv2.COLOR_RGB2BGR)
                hsv_heatmap = cv2.cvtColor(heatmap_BGR, cv2.COLOR_BGR2HSV)

                # create blended img (original + heatmap)
                blend = org * 0.6 + jet_map * 0.4
                # convert attention map to GRAY
                gray_heatmap = cv2.cvtColor(vis_map, cv2.COLOR_RGB2GRAY)

                # NiBlack Thresholding

                thr_gray_heatmap = cv2.ximgproc.niBlackThreshold(gray_heatmap, 255, cv2.THRESH_BINARY, 225, 0.7)

                try:
                    _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_NONE)
                except:
                    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

                blend_bbox = org.copy()
                wow = blend.copy()
                '''
                cv2.drawContours(wow, contours, -1, (0,255,0), 2)
                cv2.drawContours(blend_bbox, contours, -1, (0,255,0), 2)

                pts= contours
                pts = pts.reshape((-1,1,2))
                cv2.polylines(blend_bbox, contours, True, (0,255,0))


                file_name = "output/mask/myfile_"+ str(count) + ".txt"
                file = open(file_name,"w")


                for cnt in contours:
                    file.write(str(cnt)+"\n")
                    file.write("--\n")

                file.close()
                '''
                out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count))
                cv2.imwrite(out_path, vis_map)
                out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
                cv2.imwrite(out_path, v_img)

                out_path = path.join(out_dir, 'concat', '{0:06d}_concat.png'.format(count))
               # adl = cv2.imread("../ADL_WSOL/Pytorch/log_archive/0.8_epoch300/results/"+str(epoch)+"/HEAT_TEST"+str(epoch+1)+"_"+ str(count)+"_blend.jpg")
                c1 =  np.concatenate((v_img, wow), axis=1)
                c2 =  np.concatenate((c1, blend_bbox), axis=1)
                #c3 = np.concatenate((c2, adl), axis=1)
                cv2.imwrite(out_path, c2)

                count += 1

            # measure accuracy and record loss
            prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 1))
            top3_list = outputs.data.topk(1, 1, True, True)
            sample_idx = 0

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top3.update(prec3.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top3: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='best_checkpoint.pth.tar'):
    print("\nModel saved...")
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if not path.exists(path.join(args.checkpoint, 'output')):
        os.mkdir(path.join(args.checkpoint, 'output'))

    if is_best:
        if not path.exists(path.join(args.checkpoint, 'output')):
            os.mkdir(path.join(args.checkpoint, 'output'))
#        else:
        shutil.rmtree(os.path.join(args.checkpoint, 'output'))
#            os.mkdir(path.join(args.checkpoint, 'output'))

        shutil.copytree("output/", os.path.join(checkpoint, 'output'))


if __name__ == '__main__':
    main()




