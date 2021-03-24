import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, True)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n, True)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n, True)

        self.attention = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.Conv2d(self.inplanes, (int) (self.inplanes/3), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d((int)(self.inplanes/3)),
            nn.ReLU(inplace=True),
            nn.Conv2d((int)(self.inplanes/3), self.inplanes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
        )

        self.softmax = nn.Softmax(-1)
        self.bn =  nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, d_sample):
        layers = []
        if not d_sample:
            inplanes = self.inplanes

        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            # layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            if d_sample:
                layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
                self.inplanes += self.growthRate
            elif not d_sample:
                layers.append(block(inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
                inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
        input = x

        # feature extractor
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        ax = (self.dense3(x))
        ex = ax

        # resize input
        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray, (8, 8), mode='bilinear')

        # feature * image (before attention cal.)
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
        rx = self.bn(rx)
        rx = self.relu(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)

        return rx, att


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
