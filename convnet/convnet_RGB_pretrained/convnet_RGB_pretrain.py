# convnet_RGB_pretrain.py
# Pré-entraînement du ConvNet sur le Wheat Leaf Dataset
# 3 classes : Healthy, Septoria, Stripe_Rust
# Les poids seront ensuite utilisés pour fine-tuner sur notre dataset Kaggle

### IMPORTS

import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import random
import time
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix


### SEED

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


### FONCTIONS


def import_images_wheat(path):
    """
    Charge les images du Wheat Leaf Dataset.
    3 classes : Healthy, Septoria, Stripe_Rust
    Redimensionne toutes les images en 64x64 pour correspondre
    à notre architecture ConvNet.
    """
    images = []
    labels = []

    classes = {
        "Healthy":     (0, "loh", 102),
        "septoria":    (1, "los", 97),
        "stripe_rust": (2, "lolr", 208)
    }

    for folder, (label, prefix, n) in classes.items():
        for i in range(1, n + 1):
            img_path = f"{path}/{folder}/{prefix}({i}).JPG"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)

    return np.array(images), labels


def split_data(images, labels, n_val=0.1, n_test=0.2):
    """
    Split aléatoire des données en train/val/test
    par classe pour garantir l'équilibre.
    """
    train_images, val_images, test_images = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for class_idx in range(3):
        # indices de cette classe
        idx = [i for i, l in enumerate(labels) if l == class_idx]
        random.shuffle(idx)

        n = len(idx)
        n_test_c = int(n * n_test)
        n_val_c  = int(n * n_val)

        test_idx  = idx[:n_test_c]
        val_idx   = idx[n_test_c:n_test_c + n_val_c]
        train_idx = idx[n_test_c + n_val_c:]

        train_images.extend([images[i] for i in train_idx])
        val_images.extend([images[i]   for i in val_idx])
        test_images.extend([images[i]  for i in test_idx])
        train_labels.extend([labels[i] for i in train_idx])
        val_labels.extend([labels[i]   for i in val_idx])
        test_labels.extend([labels[i]  for i in test_idx])

    return (np.array(train_images), train_labels,
            np.array(val_images),   val_labels,
            np.array(test_images),  test_labels)


### TRANSFORMATIONS

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


### CLASSE DATASET

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


### ARCHITECTURE CONVNET (identique à convnet_RGB.py)

class ConvNet(nn.Module):
    """
    ConvNet from scratch adapté aux images RGB 64x64.
    3 blocs convolutifs + couches fully connected.
    """
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu    = nn.ReLU()

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
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


### CHARGEMENT DES DONNEES

path = "/home/mona/wheat_leaf"

print("Chargement des images")
images, labels = import_images_wheat(path)
print(f"Total images chargées : {len(images)}")
print(f"Healthy : {labels.count(0)}, Septoria : {labels.count(1)}, Stripe Rust : {labels.count(2)}")

# Split par classe
train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)

print(f"Train : {len(train_images)} images")
print(f"Val   : {len(val_images)} images")
print(f"Test  : {len(test_images)} images")


### HYPERPARAMETRES

batch_size    = 32
num_epochs    = 100
learning_rate = 0.001


### DATASETS ET DATALOADERS

train_dataset = CustomImageDataset(train_images, train_labels, transform=train_transform)
val_dataset   = CustomImageDataset(val_images,   val_labels,   transform=eval_transform)
test_dataset  = CustomImageDataset(test_images,  test_labels,  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


### INITIALISATION DU MODELE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

model     = ConvNet(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)


### BOUCLE D'ENTRAINEMENT

train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_val_loss            = float('inf')
epochs_sans_amelioration = 0
patience                 = 15
stop_training            = False

os.makedirs("saved_models", exist_ok=True)

print("Début du pré-entraînement.")
for epoch in range(1, num_epochs + 1):

    if stop_training:
        continue

    start_time = time.time()

    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for images_batch, labels_batch in train_loader:
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        labels_batch = labels_batch.squeeze()

        optimizer.zero_grad()
        outputs = model(images_batch)
        loss    = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * images_batch.size(0)
        _, predicted   = torch.max(outputs, 1)
        total_train   += labels_batch.size(0)
        correct_train += (predicted == labels_batch).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc  = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_val_loss = 0.0
    correct_val      = 0
    total_val        = 0

    with torch.no_grad():
        for images_batch, labels_batch in val_loader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            labels_batch = labels_batch.squeeze()
            outputs      = model(images_batch)
            loss_val     = criterion(outputs, labels_batch)

            running_val_loss += loss_val.item() * images_batch.size(0)
            _, predicted      = torch.max(outputs, 1)
            total_val        += labels_batch.size(0)
            correct_val      += (predicted == labels_batch).sum().item()

    val_loss = running_val_loss / len(val_loader.dataset)
    val_acc  = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    end_time   = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch [{epoch:02d}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Temps: {epoch_time:.2f}s")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss            = val_loss
        epochs_sans_amelioration = 0
        torch.save(model.state_dict(), "saved_models/convnet_wheat_pretrained.pth")
        print(f"  -> Meilleur modèle sauvegardé à l'epoch {epoch} (Val Loss: {val_loss:.4f})")
    else:
        epochs_sans_amelioration += 1
        if epochs_sans_amelioration >= patience:
            print(f"Early stopping à l'epoch {epoch}")
            stop_training = True


### EVALUATION SUR LE JEU DE TEST

print("\nChargement du meilleur modèle.")
model.load_state_dict(torch.load("saved_models/convnet_wheat_pretrained.pth"))

print("Evaluation sur le jeu de test Wheat Leaf.")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images_batch, labels_batch in test_loader:
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.squeeze()
        outputs      = model(images_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)
print(f"F1-score macro (Wheat Leaf) : {f1:.4f}")
print("Matrice de confusion :")
print(cm)


### COURBES

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses,   label='Val Loss',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - Pré-entraînement Wheat Leaf")
plt.legend()
plt.savefig("loss_pretrain_wheat.png")

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accuracies,   label='Val Accuracy',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Courbe d'accuracy - Pré-entraînement Wheat Leaf")
plt.legend()
plt.savefig("accuracy_pretrain_wheat.png")

print("\nPré-entraînement terminé")
print("Lancer convnet_RGB_finetuned.py pour le fine-tuning")