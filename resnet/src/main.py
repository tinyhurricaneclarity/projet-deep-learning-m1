import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd
from pathlib import Path
import skimage as ski

from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# Import des données : 600 images train

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
path_train_val = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/val/"

x_train = [] #liste qui stocke les images
y_train = [] #liste qui stocke les labels


for i in range(1,201,1):
    path_health = f"{path_train_rgb}/Health_hyper_{i}.png"
    path_other = f"{path_train_rgb}/Other_hyper_{i}.png"
    path_rust = f"{path_train_rgb}/Rust_hyper_{i}.png"
    x_train.append(ski.io.imread(path_health))
    x_train.append(ski.io.imread(path_other))
    x_train.append(ski.io.imread(path_rust))
    y_train.append(0) #"health"
    y_train.append(1) #"other"
    y_train.append(2) #"rust"


print("Nombre images train:", len(y_train)) #600 normalement

# Import des images tests rgb + labels

path_test_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/RGB/"

x_test = []

for rgb in sorted(Path(path_test_rgb).glob("*.png")): #prends toutes les images qui se terminent par rgb. glob ne marche que sur les objet Path
    x_test.append((ski.io.imread(rgb)))  # Ajoute l'image'

print("Nombre images test:", len(x_test)) #300

#Visualiser les images

labels = ["Health", "Other", "Rust"]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()): #axes.flatten() → transforme la grille en liste : [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
  img = x_train[i]
  ax.imshow(img)
  ax.set_title(labels[y_train[i]])


plt.tight_layout() #pour ajuster les tailles des images et éviter qu'elles se chevauchent
plt.show()