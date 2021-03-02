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
from os import path

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, config

from tensorboardX import SummaryWriter
from datetime import datetime
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
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_classes', type=int, default=2)

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

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--board-path', '--bp', default='board', type=str,
                    help='tensorboardx path')
parser.add_argument('--board-tag', '--tg', default='tag', type=str,
                    help='tensorboardx writer tag')

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Torch Seed
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_epoch = 0

board_time = datetime.now().isoformat()
writer_train = SummaryWriter(
    log_dir=os.path.join(args.board_path, args.dataset, "{}{:d}-bs{:d}-lr{:.5f}-wd{:.6f}-{}".format(args.arch,
                                                                                                 args.depth,
                                                                                                 args.train_batch,
                                                                                                 args.lr,
                                                                                                 args.weight_decay,
                                                                                                 args.board_tag),
                         board_time, "train"))
writer_test = SummaryWriter(
    log_dir=os.path.join(args.board_path, args.dataset, "{}{:d}-bs{:d}-lr{:.5f}-wd{:.6f}-{}".format(args.arch,
                                                                                                 args.depth,
                                                                                                 args.train_batch,
                                                                                                 args.lr,
                                                                                                 args.weight_decay,
                                                                                                 args.board_tag),
                         board_time, "test"))


def main():
    global best_acc, best_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    args.checkpoint = os.path.join(args.checkpoint, board_time)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    print('==> Preparing dataset')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.STL10
    args.num_classes = 10

    trainset = dataloader(root='./data', split='train', download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    valset = dataloader(root='./data', split='test', download=True, transform=transform_test)
    val_loader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    ## create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
        )

    # training from scratch or during eval
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.num_classes)

        if not args.evaluate:
            print("=> Training from scratch ...")
        else:
            print("=> Evaluating Mode...")
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ## Resume training or for eval mode
    title = 'ImageNet-' + args.arch

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        resume_folder = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, args.epochs, use_cuda)

        if not path.exists(path.join(args.checkpoint, 'output')):
            os.mkdir(path.join(args.checkpoint, 'output'))
        shutil.rmtree(os.path.join(args.checkpoint, 'output'))
        shutil.copytree("output/", os.path.join(args.checkpoint, 'output'))

        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = (test_acc > best_acc)

        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1

        print("Best acc: %f , Epoch: %d" % (best_acc, best_epoch))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print("Best acc: %f , Epoch: %d" % (best_acc, best_epoch))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top2=top2.avg
        )
        n_iter = epoch * len(trainloader) + batch_idx + 1
        writer_train.add_scalar('Train/loss', loss.data.item(), n_iter)
        writer_train.add_scalar('Train/top1', prec1.data.item(), n_iter)
        writer_train.add_scalar('Train/top5', prec5.data.item(), n_iter)

        bar.next()

    writer_train.add_scalar('Avg.loss', losses.avg, epoch)
    writer_train.add_scalar('Avg.top1', top1.avg, epoch)
    writer_train.add_scalar('Avg.top5', top5.avg, epoch)

    # for name, param in model.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer_train.add_histogram("{}/{}".format(layer, attr), param, epoch)

    return (losses.avg, top1.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    softmax = nn.Softmax()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

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
            # attention, fe, per = attention

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

                if not path.exists(path.join(out_dir, 'concat')):
                    os.mkdir(path.join(out_dir, 'concat'))

                v_img = ((item_img.transpose((1, 2, 0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225]) * 256
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
                vis_map = vis_map / np.max(vis_map)
                vis_map = np.uint8(256 * vis_map)

                jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

                # create blended img (original + heatmap)
                blend = org * 0.6 + jet_map * 0.4

                out_path = path.join(out_dir, 'concat', '{0:06d}_concat.png'.format(count))
                c1 = np.concatenate((v_img, blend), axis=1)
                cv2.imwrite(out_path, c1)

                if args.evaluate:
                    attpath = os.path.join(os.path.dirname(args.resume), 'att')
                    if not path.exists(attpath):
                        os.mkdir(attpath)
                    # np.save(os.path.join(attpath, "{0:06d}".format(count)), resize_att)
                    np.save(os.path.join(attpath, "{0:06d}".format(count)), vis_map)
                count += 1

            # measure accuracy and record loss
            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top2.update(prec2.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f}'.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top2=top2.avg,
            )
            n_iter = epoch * len(val_loader) + batch_idx + 1
            writer_test.add_scalar('Test/loss', loss.data.item(), n_iter)
            writer_test.add_scalar('Test/top1', prec1.data.item(), n_iter)
            writer_test.add_scalar('Test/top5', prec2.data.item(), n_iter)
            bar.next()

        writer_test.add_scalar('Avg.loss', losses.avg, epoch)
        writer_test.add_scalar('Avg.top1', top1.avg, epoch)
        writer_test.add_scalar('Avg.top5', top2.avg, epoch)
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    print("\nModel saved...")
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        print("\nBEST Model updated...")
        if not path.exists(path.join(args.checkpoint, 'output')):
            os.mkdir(path.join(args.checkpoint, 'output'))
        shutil.rmtree(os.path.join(args.checkpoint, 'output'))
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        shutil.copytree("output/", os.path.join(checkpoint, 'output'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
