# src/models.py
"""
Tous les modèles pour la classification de la rouille jaune.
Contient : ConvNet, ResNet18, ResNet50, ResNeXt50
"""

import torch
import torch.nn as nn


# ============================================================================
# CONVNET
# ============================================================================

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


# ============================================================================
# RESNET18
# ============================================================================

class BasicBlock(nn.Module): 
    """Bloc résiduel basique pour ResNet18"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        # Dans le cas où l'input n'est pas de la même taille que l'output
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.in_channels = 64
        
        # Entrée du réseau d'après He et al 2016
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 blocs du ResNet
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================================
# RESNET50
# ============================================================================

class BottleneckBlock(nn.Module):
    """Bloc bottleneck pour ResNet50"""
    expansion = 4

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

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.in_channels = 64

        # Entrée du réseau
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Blocs bottleneck ResNet50
        self.layer1 = self._make_layer(BottleneckBlock, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================================
# RESNEXT50
# ============================================================================

class ResNeXtBlock(nn.Module):
    """Bloc ResNeXt avec cardinality"""
    def __init__(self, in_channels, cardinality, bwidth, idt_downsample=None, stride=1):
        super(ResNeXtBlock, self).__init__()
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


class ResNeXt50(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, cardinality=32, bwidth=4):
        super(ResNeXt50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.cardinality = cardinality
        self.bwidth = bwidth
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNeXt Layers
        self.layer1 = self._make_layer(ResNeXtBlock, 3, stride=1)
        self.layer2 = self._make_layer(ResNeXtBlock, 4, stride=2)
        self.layer3 = self._make_layer(ResNeXtBlock, 6, stride=2)
        self.layer4 = self._make_layer(ResNeXtBlock, 3, stride=2)
        
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

    def _make_layer(self, resnext_block, no_residual_blocks, stride):
        identity_downsample = None
        out_channels = self.cardinality * self.bwidth
        layers = []

        if stride != 1 or self.in_channels != out_channels * 2:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*2)
            )

        layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth, identity_downsample, stride))
        self.in_channels = out_channels * 2

        for i in range(no_residual_blocks - 1):
            layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth))

        self.bwidth *= 2

        return nn.Sequential(*layers)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_model(model_name, in_channels=3, num_classes=3, input_size=64):
    """
    Factory function pour créer n'importe quel modèle
    
    Args:
        model_name: 'convnet', 'resnet18', 'resnet50', 'resnext50'
        in_channels: nombre de canaux d'entrée
        num_classes: nombre de classes
        input_size: taille spatiale (utilisé uniquement pour ConvNet)
    
    Returns:
        model: PyTorch model
    """
    models_dict = {
        'convnet': lambda: ConvNet(in_channels, input_size, num_classes),
        'resnet18': lambda: ResNet18(in_channels, num_classes),
        'resnet50': lambda: ResNet50(in_channels, num_classes),
        'resnext50': lambda: ResNeXt50(in_channels, num_classes),
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not recognized. Choose from {list(models_dict.keys())}")
    
    return models_dict[model_name]()


def get_input_channels(modality):
    """Retourne le nombre de canaux d'entrée selon la modalité"""
    channels_map = {
        'RGB': 3,
        'MS': 5,
        'HS': 125,
        'RGB_MS': 8,
        'RGB_MS_HS': 133,
        'MS_HS': 130,
        'MS_sans_other': 5,
        'HS_sans_other': 125,
    }
    
    if modality not in channels_map:
        raise ValueError(f"Modality {modality} not recognized. Choose from {list(channels_map.keys())}")
    
    return channels_map[modality]


def get_input_size(modality):
    """Retourne la taille spatiale selon la modalité"""
    size_map = {
        'RGB': 64,
        'MS': 64,
        'HS': 32,
        'RGB_MS': 64,
        'RGB_MS_HS': 64,
        'MS_HS': 64,
        'MS_sans_other': 64,
        'HS_sans_other': 32,
    }
    
    if modality not in size_map:
        raise ValueError(f"Modality {modality} not recognized")
    
    return size_map[modality]