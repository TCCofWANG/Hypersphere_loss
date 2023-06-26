# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:44:34 2021

Paper:Spherenet: Learning spherical representations for detection and classification in omnidirectional
"""
import math,torch
import torch.nn as nn

__all__ = ['SphereFace4', 'SphereFace10', 'SphereFace20', 'SphereFace36', 'SphereFace64']

cfg = {
    'A': [0, 0, 0, 0],
    'B': [0, 1, 2, 0],
    'C': [1, 2, 4, 1],
    'D': [2, 4, 8, 2],
    'E': [3, 8, 16, 3]
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    c3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)    
    nn.init.normal_(c3x3.weight, 0, 0.01)
    return c3x3


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    c1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    nn.init.normal_(c1x1.weight, 0, 0.01)
    return c1x1


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SphereFace(nn.Module):
    """
    Implement paper which is 'SphereFace: Deep Hypersphere Embedding for Face Recognition'.

    Reference:
        https://arxiv.org/abs/1704.08063
    """

    def __init__(self, block, layers, feat_dim=128):
        """

        :param block: residual units.
        :param layers: number of repetitions per residual unit.
        :param num_classes:
        """
        super(SphereFace, self).__init__()
        self.conv1 = conv3x3(3, 64, 2)        
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)

        self.conv2 = conv3x3(64, 128, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU(128)

        self.conv3 = conv3x3(128, 256, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.prelu3 = nn.PReLU(256)

        self.conv4 = conv3x3(256, 512, 2)
        self.bn4 = nn.BatchNorm2d(512)
        self.prelu4 = nn.PReLU(512)
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        
        self.fc5 = nn.Sequential(nn.Linear(512*7*6, feat_dim),
                                 nn.BatchNorm1d(feat_dim),
                                 nn.PReLU(num_parameters = feat_dim),
                                 )
        
             
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)                
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks != 0:
            layers = []
            for _ in range(0, blocks):
                downsample = nn.Sequential(
                    conv1x1(planes, planes, stride),
                    nn.BatchNorm2d(planes),
                )
                nn.init.constant_(downsample[1].weight, 1)
                nn.init.constant_(downsample[1].bias, 0)
                layers.append(block(planes, planes, stride, downsample))

            return nn.Sequential(*layers)
        else:
            return None

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        if self.layer1 is not None:
            x = self.layer1(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        if self.layer2 is not None:
            x = self.layer2(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        if self.layer3 is not None:
            x = self.layer3(x)
        x = self.prelu4(self.bn4(self.conv4(x)))
        if self.layer4 is not None:
            x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return x
    
def SphereFace4(**kwargs):
    """
    Constructs a SphereFace-4 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['A'], **kwargs)

    return model


def SphereFace10(**kwargs):
    """
    Constructs a SphereFace-10 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['B'], **kwargs)

    return model

def SphereFace20(**kwargs):
    """
    Constructs a SphereFace-20 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['C'], **kwargs)

    return model


def SphereFace36(**kwargs):
    """
    Constructs a SphereFace-36 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['D'], **kwargs)

    return model


def SphereFace64(**kwargs):
    """
    Constructs a SphereFace-64 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['E'], **kwargs)

    return model

#if __name__=='__main__':
#    net=SphereFace20(feat_dim=2)
#    y=net(torch.randn(2,3,112,96))
#    
##    print(net)
#    print(y.data.numpy())