import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path
from skimage.transform import resize
from collections import Counter

#machine learing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optim


class resnext_block(nn.Module):

  def __init__(self, in_channels, cardinality, bwidth, idt_downsample=None, stride=1):
    super(resnext_block, self).__init__()
    self.expansion = 2
    out_channels = cardinality * bwidth
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=cardinality, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = idt_downsample

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
        identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

#ResNext :
class ResNeXt(nn.Module):

  def __init__(self, resnet_block, layers, cardinality, bwidth, img_channels, num_classes):
    super(ResNeXt, self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.cardinality = cardinality
    self.bwidth = bwidth
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # ResNeXt Layers
    self.layer1 = self._layers(resnext_block, layers[0], stride=1)
    self.layer2 = self._layers(resnext_block, layers[1], stride=2)
    self.layer3 = self._layers(resnext_block, layers[2], stride=2)
    self.layer4 = self._layers(resnext_block, layers[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(self.cardinality * self.bwidth, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x

  def _layers(self, resnext_block, no_residual_blocks, stride):
    identity_downsample = None
    out_channels = self.cardinality * self.bwidth
    layers = []

    if stride != 1 or self.in_channels != out_channels * 2:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*2, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*2))

      layers.append(resnext_block(self.in_channels,  self.cardinality, self.bwidth, identity_downsample, stride))
    self.in_channels = out_channels * 2

    for i in range(no_residual_blocks - 1):
      layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth))

    self.bwidth *= 2

    return nn.Sequential(*layers)

  def _layers(self, resnext_block, no_residual_blocks, stride):
        identity_downsample = None
        out_channels = self.cardinality * self.bwidth
        layers = []

        if stride != 1 or self.in_channels != out_channels * 2:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*2, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*2))

        layers.append(resnext_block(self.in_channels,  self.cardinality, self.bwidth, identity_downsample, stride))
        self.in_channels = out_channels * 2

        for i in range(no_residual_blocks - 1):
            layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth))

        self.bwidth *= 2

        return nn.Sequential(*layers)

#ResNext50 :
def ResNeXt50(img_channels=3, num_classes=1000, cardinality=32, bwidth=4):
    return ResNeXt(resnext_block, [3,4,6,3],  cardinality, bwidth, img_channels, num_classes)

