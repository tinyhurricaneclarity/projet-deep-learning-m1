#Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Bloc résiduel resnet18 de geeksforgeeks

class BasicBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1): #stride
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() #par défaut ne fait rien

        #dans le cas ou l'input n'est pas de la meme taille que l'output, on applique la séquence (sequential) suivante :
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
    
"""Pour la structure des Resnet voir l'article de référence de He et al., 2016"""

class ResNet18(nn.Module): #de geeksforgeeks
    def __init__(self, num_classes=3): #num classes c'est le nombre de classes que va prédire le modèle 
        super().__init__() #super permet d'initialiser la classe parent : nn.Module
        self.in_channels = 64 #nombre de canaux d'entrée

        #entrée du réseau
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #input, output (64 features), taille kernel
        self.bn1 = nn.BatchNorm2d(64) #noramlisation
        self.relu = nn.ReLU(inplace=True) #fonction activation non linéaire


        #4 blocs du ResNet : chaque layer correspond à un groupe de bloc résiduel
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 10
train_losses, train_acc_list, test_acc_list = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    test_acc_list.append(test_acc)
    
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')