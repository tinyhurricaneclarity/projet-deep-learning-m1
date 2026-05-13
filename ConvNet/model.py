# model.py
# Architecture du ConvNet adapté aux images du dataset

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    ConvNet from scratch adapté aux petites images du dataset.
    
    Architecture :
    - 3 blocs convolutifs (Conv2d + BatchNorm2d + ReLU + MaxPool2d)
    - 2 couches fully connected
    - Dropout(0.3) pour réduire le surapprentissage
    
    Paramètres :
    - in_channels : nombre de canaux en entrée (3=RGB, 5=MS, 125=HS, ...)
    - input_size  : taille spatiale des images en entrée (64 pour RGB/MS, 32 pour HS)
    - num_classes : nombre de classes (3 par défaut : Health, Rust, Other)
    """

    def __init__(self, in_channels=3, input_size=64, num_classes=3):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu    = nn.ReLU()

        # Calcul automatique de la taille après 3 maxpoolings
        # 64x64 → 8x8 / 32x32 → 4x4
        fc_input_size = 128 * (input_size // 8) * (input_size // 8)

        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x