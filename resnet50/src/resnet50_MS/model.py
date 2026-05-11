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
    expansion = 4 #facteur de 4, càd que le out channels est le in channels multiplié par 4. ATTENTION à ne pas confondre channels et taille de l'images

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity #skip connection
        out = self.relu(out)

        return out #retourne le résultat du bloc résiduel bottleneck


class ResNet50(nn.Module): 
    def __init__(self, num_classes=3): #num classes c'est le nombre de classes que va prédire le modèle 
        super().__init__() #super permet d'initialiser la classe parent : nn.Module
        self.in_channels = 64 #nombre de canaux attendus en entrée du premier bloc résiduel

        ##entrée du réseau d'après he et al 2016
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #input, output (64 features), taille kernel. padding 3 pour avoir 112x112 en sortie
        self.bn1 = nn.BatchNorm2d(64) #noramlisation
        self.relu = nn.ReLU(inplace=True) #fonction activation non linéaire
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #prend la valeur max dans un kernel de 3x3, pas de 2. La taille de l'image est divisée par deux.

        #Blocs bottleneck ResNet50 : chaque layer correspond à un groupe de bloc résiduel
        self.layer1 = self._make_layer(BottleneckBlock, 64, num_blocks=3, stride=1) #autre version existe voir le site cité (medium)
        self.layer2 = self._make_layer(BottleneckBlock, 128, num_blocks=4, stride=2) #stride=2 divise la taille de l'image par deux
        self.layer3 = self._make_layer(BottleneckBlock, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #résume l'image en un vecteur
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes) #fc : Fully Connected : classification. Applique une fonction linéaire qui permet de classifier. (512 * 4 = 2048)  est le nombre de canaux de la dernière couche

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None #on crée un downsample dès que les canaux ou la résolution changent.
        if stride != 1 or self.in_channels != out_channels * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BottleneckBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BottleneckBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1) #flatten (tensor → vecteur)
        out = self.fc(out)
        return out





