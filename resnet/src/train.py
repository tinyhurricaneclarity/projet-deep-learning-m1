#Import des packages

import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import skimage as ski

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import data_load
import model


#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet18().to(device) #lance une nouvelle instance du modèle, ca réinitialise tout.
print(model)

#Suivi des paramètres avec un dictionnaire (pour pouvoir les utiliser en paramètre)

#Syntaxe pour futur grid search

config_counter = 1
num_epochs = 100
learning_rate = 0.1
batch_size = 32
momentum = 0.9
weight_decay = 5e-4
step_size = 30
gamma = 0.1
criterion_config = "CrossEntropy"
optimizer_config = "SGD"
scheduler_config =  "StepLR"

config = {
    "config_counter": config_counter,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "step_size": step_size,
    "gamma": gamma,
    "criterion": criterion_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
  
}


#Choix de la fonction d'activation, de l'optimizer et du scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])


#Import des données train provenant de data.py

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
x_train, y_train = data_load.load_data_train(path_train_rgb)

#Convertion en tensor et trainloader
dataset = data_load.CustomImageDataset(x_train, y_train, transform=None)

#Dataloader
train_loader, val_loader, test_loader = data_load.create_dataloader(dataset, batch_size=batch_size)
print(len(train_loader), len(val_loader))

#Liste pour stocker les résultats (loss, acc)
train_losses, train_acc_list, val_losses,  val_acc_list = [], [], [], []


#entrainement du modèle

for epoch in range(num_epochs):
    model.train()
    train_running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = train_running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

#évaluation des train et val

    model.eval()
    correct, total = 0, 0
    val_running_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    val_losses.append(val_loss)
    val_acc_list.append(val_acc)
    
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
    

#affichage des résultats et des paramètres

config_str = f"config{config_counter}, optimizer{optimizer_config}, epochs{num_epochs}, lr{learning_rate}, batch_size{batch_size}, momentum{momentum}, weight_decay{weight_decay}, step_size{step_size}, gamma{gamma}"
print(f"Running config {config_str}")


plt.figure(figsize=(12,5))

plt.suptitle(config_str, fontsize=8)
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Resnet18 - Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Resnet18 - Accuracy')
plt.legend()

plt.show()