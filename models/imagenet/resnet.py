import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=1000, dropout=0.5):
        self.inplanes = 64
        self.dropout= dropout

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)


        self.att_conv = self._make_layer(block, 512, layers[3], stride=1, down_size=False)
        self.att_bn = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv2   = nn.Conv2d(512 * block.expansion, 1024, kernel_size=1, padding=0,
                               bias=False)
        self.att_bn2 = nn.BatchNorm2d(1024)
        #self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
        #                       bias=False)
        #self.att_conv3  = nn.Conv2d(num_classes,1, kernel_size=3, padding=1,
        #                       bias=False)
        #self.bn_att3 = nn.BatchNorm2d(1)
        #self.att_gap = nn.AvgPool2d(14)
        #self.sigmoid = nn.Sigmoid()

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Sequential(nn.Dropout(p= self.dropout), nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

            return nn.Sequential(*layers)


    def forward(self, x):
        # original input size (32, 3, 224, 224)
        input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        fe = x # feature map size (32, 1024, 14, 14)

        #_, C, _, _ = fe.shape
        # normalization
        #act_min, _ = fe.view(1, C, -1).min(dim=2)
        #act_min = act_min.view(1, C, 1, 1)
        #act_max, _ = fe.view(1, C, -1).max(dim=2)
        #act_max = act_max.view(1, C, 1, 1)
        #denominator = torch.where(
        #        (act_max - act_min) != 0., act_max - act_min, torch.tensor(1.).cuda())
        #fe = fe / denominator

        # resize input to (32, 1, 14, 14)
        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray,(14, 14), mode='bilinear')

        # new feature map
        new_fe = input_resized * fe

        # feature importance extractor w/ residual blocks
        '''
        self.att_conv   = nn.Conv2d(1024, 2048, kernel_size=1, padding=0,
                               bias=False).cuda()
        self.att_bn = nn.BatchNorm2d(2048).cuda()
        self.att_conv1   = nn.Conv2d(2048, 1024, kernel_size=1, padding=0,
                               bias=False).cuda()
        self.att_bn1 = nn.BatchNorm2d(1024).cuda()
        self.att_conv2   = nn.Conv2d(1024, 1024, kernel_size=1, padding=0,
                               bias=False).cuda()
        self.att_bn2 = nn.BatchNorm2d(1024).cuda()
        '''


        ax = self.att_conv(new_fe)
        ax = self.att_bn(ax)
        ax = self.att_conv2(ax)
        ax = self.att_bn2(ax)


        w = F.softmax(ax) # weight (1024 channel)

        # weighted sum
        att = w * fe

        # attention mechansim
        rx = att * fe
        rx = rx + fe
        per = rx

        # classifier
        rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)

        '''

        ax = self.bn_att(self.att_layer4(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        #print(ax.shape)
        px =ax
        bs, cs, ys, xs = ax.shape
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        #print("att",self.att.shape)
        #self.att = self.att.view(bs, 1, ys, xs)
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        #print(x.shape)
        rx = fe * self.att
        #print(self.att.shape)
        #rx = x * self.att
        #print("rx", rx.shape)
        rx = rx + x
        per = rx
        rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)
        '''

        return rx, [att, fe, per]

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


