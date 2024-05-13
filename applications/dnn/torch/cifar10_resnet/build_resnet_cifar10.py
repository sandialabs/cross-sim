import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import sys

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes,stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet_cifar10(nn.Module):

    def __init__(self, num_blocks, in_channels=3, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=0.99, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self._make_layer(16, num_blocks, stride=1)
        self.res2 = self._make_layer(32, num_blocks, stride=2)
        self.res3 = self._make_layer(64, num_blocks, stride=2)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Linear(64, num_classes))
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * BasicBlock.expansion, stride),
                self.norm_layer(planes * BasicBlock.expansion),
            )
        layers = []
        layers.append(
            BasicBlock(self.in_planes, planes*BasicBlock.expansion, stride, downsample))
        self.in_planes = planes*BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.classifier(out)
        return out