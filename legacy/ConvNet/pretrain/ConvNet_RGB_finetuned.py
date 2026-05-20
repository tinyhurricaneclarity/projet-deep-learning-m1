# convnet_RGB_finetuned.py
# Fine-tuning du ConvNet pré-entraîné sur Wheat Leaf Dataset
# sur notre dataset Kaggle (Health, Rust, Other)
# Les poids pré-entraînés sont chargés depuis pretrain/saved_models/convnet_wheat_pretrained.pth

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix

# Accès au module parent pour importer model.py, config.py, data_load.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ConvNet
from config import (
    PATH, CLASS_NAMES, BATCH_SIZE, NUM_DATA,
    SEED_RANDOM, SEED_TORCH
)
from data_load import (
    alea_train_test, sufix_and_path, import_images,
    get_transforms, CustomImageDataset
)


### SEED

random.seed(SEED_RANDOM)
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)


### CHEMINS

PRETRAIN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_models", "convnet_wheat_pretrained.pth"
)
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_models"
)
os.makedirs(SAVE_DIR, exist_ok=True)


### CHARGEMENT DES DONNEES (même pipeline que main.py)

Im_type  = "RGB"
num_classes = 3

dico_train_test = alea_train_test(NUM_DATA, CLASS_NAMES, n_val=99, n_test=99)
sufix_and_path(Im_type, dico_train_test, PATH)

Train = import_images(CLASS_NAMES, dico_train_test, "train")
Val   = import_images(CLASS_NAMES, dico_train_test, "val")
Test  = import_images(CLASS_NAMES, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")


### TRANSFORMATIONS
# Normalisation ImageNet car les poids pré-entraînés ont été appris avec cette normalisation

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


### DATASETS ET DATALOADERS

train_dataset = CustomImageDataset(Train['images'], Train['labels'], transform=train_transform)
val_dataset   = CustomImageDataset(Val['images'],   Val['labels'],   transform=eval_transform)
test_dataset  = CustomImageDataset(Test['images'],  Test['labels'],  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


### HYPERPARAMETRES

num_epochs    = 100
learning_rate = 0.001


### INITIALISATION DU MODELE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model = ConvNet(in_channels=3, input_size=64, num_classes=3).to(device)

# Chargement des poids pré-entraînés sur Wheat Leaf
model.load_state_dict(torch.load(PRETRAIN_PATH))
print(f"Poids pré-entraînés chargés depuis {PRETRAIN_PATH}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)


### BOUCLE D'ENTRAINEMENT

train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []

best_val_loss            = float('inf')
epochs_sans_amelioration = 0
patience                 = 15
stop_training            = False

print("\nDébut du fine-tuning.")
for epoch in range(1, num_epochs + 1):

    if stop_training:
        continue

    start_time = time.time()

    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze()

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        total_train   += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc  = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_val_loss = 0.0
    correct_val      = 0
    total_val        = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels         = labels.squeeze()
            outputs        = model(images)
            loss_val       = criterion(outputs, labels)

            running_val_loss += loss_val.item() * images.size(0)
            _, predicted      = torch.max(outputs, 1)
            total_val        += labels.size(0)
            correct_val      += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader.dataset)
    val_acc  = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - start_time

    print(f"Epoch [{epoch:02d}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Temps: {epoch_time:.2f}s")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss            = val_loss
        epochs_sans_amelioration = 0
        torch.save(model.state_dict(), f"{SAVE_DIR}/convnet_RGB_finetuned.pth")
        print(f"  -> Meilleur modèle sauvegardé (Val Loss: {val_loss:.4f})")
    else:
        epochs_sans_amelioration += 1
        if epochs_sans_amelioration >= patience:
            print(f"Early stopping à l'epoch {epoch}")
            stop_training = True


### EVALUATION FINALE

print("\nChargement du meilleur modèle.")
model.load_state_dict(torch.load(f"{SAVE_DIR}/convnet_RGB_finetuned.pth"))

print("Evaluation sur le jeu de test.")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images       = images.to(device)
        labels       = labels.squeeze()
        outputs      = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)
f1_par_classe = f1_score(all_labels, all_preds, average=None)

print(f"\nF1-score macro : {f1:.4f}")
print("Matrice de confusion :")
print(cm)
print("\nF1-score par classe :")
for i, classe in enumerate(CLASS_NAMES):
    print(f"  F1-score {classe} : {f1_par_classe[i]:.4f}")


### COURBES

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses,   label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - ConvNet RGB fine-tuné (Wheat Leaf → Kaggle)")
plt.legend()
plt.savefig(f"{SAVE_DIR}/loss_convnet_RGB_finetuned.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies,   label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Courbe d'accuracy - ConvNet RGB fine-tuné (Wheat Leaf → Kaggle)")
plt.legend()
plt.savefig(f"{SAVE_DIR}/accuracy_convnet_RGB_finetuned.png")
plt.close()

print(f"\nTerminé. Résultats sauvegardés dans {SAVE_DIR}/")