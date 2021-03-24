from __future__ import absolute_import
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np

# from sklearn.preprocessing import MinMaxScaler

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

        print("It's LFIv2 network!!!!!!!!")

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, down_size=True)
        self.layer2 = self._make_layer(block, 32, n, stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 64, n, stride=2, down_size=True)

        self.attention = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64 * block.expansion),
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 * block.expansion, 64* block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64 * block.expansion),
            nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
        )

        self.softmax = nn.Softmax(-1)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Linear(64 * block.expansion, num_classes)
        )


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
        input = x

        # feature extractor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        ax = self.layer3(x)
        ex = ax

        # resize input
        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray, (8, 8), mode='bilinear')

        # fe normalization & feature * image (before attention cal.)
        fe = ax.clone()
        org = fe.clone()
        a1, a2, a3, a4= fe.size()
        fe = fe.view(a1, a2, -1)

        fe -= fe.min(2, keepdim=True)[0]
        fe /= fe.max(2, keepdim=True)[0]
        fe = fe.view(a1, a2, a3,a4)

        fe[torch.isnan(fe)] = 1
        fe[(org ==0)] = 0

        new_fe = fe * input_resized

        # FIN
        ax = self.attention(new_fe)
        w = self.softmax(ax.view(ax.size(0), -1))

        b, c, u, v = fe.size()
        score_saliency_map = torch.zeros((b, 1, u, v)).cuda()

        for i in range(c):
            saliency_map = torch.unsqueeze(ex[:, i, :, :], 1)
            score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)
            score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        org = score_saliency_map.clone()
        a1, a2, a3, a4= score_saliency_map.size()
        score_saliency_map = score_saliency_map.view(a1, a2, -1)

        score_saliency_map -= score_saliency_map.min(2, keepdim=True)[0]
        score_saliency_map /= score_saliency_map.max(2, keepdim=True)[0]
        score_saliency_map = score_saliency_map.view(a1, a2, a3,a4)

        score_saliency_map[torch.isnan(score_saliency_map)] = org[torch.isnan(score_saliency_map)]

        att = score_saliency_map

        # attention mechanism
        rx = att * ex
        rx = rx + ex

        # classifier
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)
        return rx, att


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
