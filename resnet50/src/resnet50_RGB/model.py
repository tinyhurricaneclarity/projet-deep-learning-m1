#Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

    
"""Pour la structure des Resnet voir l'article de référence de He et al., 2016"""

#les structures resnet50 ont des blocks "bottleneck", ou la taille du kernel varie en fonction de la couche de convolution
#https://medium.com/data-scientists-diary/building-resnet-50-101-and-152-models-from-scratch-in-pytorch-f1e84cbafa63

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False) #ratio de 4
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        return self.relu(out)


class ResNet50(nn.Module): 
    def __init__(self, num_classes=3): #num classes c'est le nombre de classes que va prédire le modèle 
        super().__init__() #super permet d'initialiser la classe parent : nn.Module
        self.in_channels = 64 #nombre de canaux d'entrée

        #entrée du réseau
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False) #input, output (64 features), taille kernel
        self.bn1 = nn.BatchNorm2d(64) #noramlisation
        self.relu = nn.ReLU(inplace=True) #fonction activation non linéaire
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Blocs bottleneck ResNet50 : chaque layer correspond à un groupe de bloc résiduel
        self.layer1 = self._make_layer(BottleneckBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #résume l'image en un vecteur
        self.fc = nn.Linear(512, num_classes) #fc : Fully Connected : classification

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # + permet de concaténer deux listes. * permet de répéter le 1
        """
        num_blocks = 3
        [1] * (3 - 1)
        = [1] * 2
        = [1, 1] #on répète 2 fois le 1
        donc avec le + 
        [2] + [1, 1]
        = [2, 1, 1]
        

    Pourquoi on fait ça ?

    👉 parce que dans un bloc ResNet :

    le 1er bloc peut réduire la taille (stride=2)
    les autres blocs gardent la taille (stride=1)
    “le premier bloc peut downsampler, les autres non"

        """

        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1) #flatten (tensor → vecteur)
        out = self.fc(out)
        return out





