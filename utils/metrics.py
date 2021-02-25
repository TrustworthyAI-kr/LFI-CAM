import numpy as np
import argparse

from os import listdir
from os.path import isfile, join
from misc import AverageMeter
# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-b', '--att-base', default='out/att', type=str)
parser.add_argument('-t', '--att-target', default='out/att', type=str)
parser.add_argument('--threshold', default=0, type=int,
                    help='Threshold value for IoU calculation ')
args = parser.parse_args()


def main():
    base_atts = [f for f in listdir(args.att_base) if isfile(join(args.att_base, f))]
    target_atts = [f for f in listdir(args.att_target) if isfile(join(args.att_target, f))]

    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    bar = Bar('Processing', max=len(base_atts))
    count = 0
    for base, target in zip(base_atts, target_atts):
        base_att = np.load(join(args.att_base, base))
        target_att = np.load(join(args.att_target, target))

        # base_att = np.uint8(base_att)
        # target_att = np.uint8(target_att)
        base_att = (base_att > args.threshold).astype(int)
        target_att = (target_att > args.threshold).astype(int)

        # IoU calculation..
        # binary class (one-hot vector) version
        # intersection = np.logical_and(base_att, target_att)
        # union = np.logical_or(base_att, target_att)
        # iou_score = np.sum(intersection) / np.sum(union)

        axes = (1, 2)
        intersection = np.sum(np.abs(target_att * base_att), axis=axes)
        mask_sum = np.sum(np.abs(base_att), axis=axes) + np.sum(np.abs(target_att), axis=axes)
        # intersection = np.sum(np.abs(target_att * base_att))
        # mask_sum = np.sum(np.abs(base_att)) + np.sum(np.abs(target_att))
        union = mask_sum - intersection
        smooth = 0.000001
        iou_score = (intersection + smooth) / (union + smooth)
        dice_score = 2 * (intersection + smooth) / (mask_sum + smooth)
        iou_meter.update(iou_score.mean(), 1)
        dice_meter.update(dice_score.mean(), 1)
        bar.suffix = '({index}) IoU: {iou:.3f} | Dice: {dice:.3f} | Total: {total:} | ETA: {eta:}'.format(
            index=count + 1,
            iou=iou_meter.avg,
            dice=dice_meter.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        count += 1
        bar.next()
    bar.finish()

if __name__ == '__main__':
    main()