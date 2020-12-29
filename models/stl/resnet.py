from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=1,stride= 3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, down_size=True)
        self.layer2 = self._make_layer(block, 32, n, stride=2, down_size=True)

        self.block1 = self._make_layer(block, 32, n, stride=1, down_size=False)
        self.block2 = self._make_layer(block, 64, n, stride=1, down_size=True)
        self.block3 = self._make_layer(block, 64, n, stride=1, down_size=False)

        self.att_conv1 = self._make_layer(block, 64, n, stride=1, down_size=False)
        self.att_bn1 = nn.BatchNorm2d(64* block.expansion)
        self.att_conv2 = self._make_layer(block, 128, n, stride=1, down_size=True)
        self.att_bn2 = nn.BatchNorm2d(128* block.expansion)
        self.att_conv3 = self._make_layer(block, 64, n, stride=1, down_size=False)
        self.att_bn3 = nn.BatchNorm2d(64* block.expansion)

        #self.layer3 = self._make_layer(block, 32, n, stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(9)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        #self.fc = nn.Sequential(nn.Dropout(p= 0.5), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))
            ##print(nn.Sequential(*layers))
            return nn.Sequential(*layers)


    def forward(self, x):
        input =x

        # feature extractor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32


        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16 (cifar)

        # resize input
        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray,(16, 16), mode='bilinear')

        # block 1
        ax = self.block1(x)
        ex =ax

        # block 2, 3
        ax = self.block2(ax)
        ax = self.block3(ax)

        fe = ax
        new_fe = fe * input_resized

        # feature importance extractor
        ax = self.att_conv1(new_fe)
        ax = self.att_bn1(ax)
        ax = self.att_conv2(ax)
        ax = self.att_bn2(ax)
        ax = self.att_conv3(ax)
        ax = self.att_bn3(ax)

        ax = self.avgpool(ax)
        bx = ax.view(ax.size(0), -1)
        w = F.softmax(bx)

        b, c, u, v = fe.size()
        score_saliency_map= torch.zeros((b,1,u,v)).cuda()

        for i in range(c):
            saliency_map = torch.unsqueeze(fe[:, i,:,:], 1)

            #if saliency_map.max() == saliency_map.min():
            #    continue

            # norm
            #norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:,i],1),1),1)
            score_saliency_map += score* saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        att= score_saliency_map

        # attention mechansim
        rx = att * ex
        rx = rx + ex

        # classifier
        rx = self.block2(rx)
        rx = self.block3(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)

        return rx, att, w


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


