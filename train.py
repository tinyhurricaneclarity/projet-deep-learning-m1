# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import config
from dataset import RGBDataset, load_data, split_data, transform

# ---- Reproductibilité ----
torch.manual_seed(config.SEED)

# ---- Chargement des données ----
print("Chargement des données...")
image_paths, labels = load_data(config.ROOT, config.TRAIN_DIR)
train_paths, train_labels, test_paths, test_labels = split_data(image_paths, labels)

# Split train en train + validation (80% train, 20% val)
train_dataset_full = RGBDataset(train_paths, train_labels, transform=transform)
val_size  = int(0.2 * len(train_dataset_full))
train_size = len(train_dataset_full) - val_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

test_dataset = RGBDataset(test_paths, test_labels, transform=transform)

# Création des dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=config.NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

print(f"Train : {train_size} | Val : {val_size} | Test : {len(test_dataset)}")

# ---- Modèle AlexNet ----
print("Chargement d'AlexNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# Charge AlexNet pré-entraîné (syntaxe correcte sans warning)
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Gèle les couches convolutives (transfer learning)
for param in model.features.parameters():
    param.requires_grad = False

# Remplace la dernière couche pour avoir 3 classes au lieu de 1000
model.classifier[6] = nn.Linear(4096, len(config.LABELS))
model = model.to(device)

# ---- Entraînement ----
criterion = nn.CrossEntropyLoss()
# On optimise seulement les paramètres non gelés
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR
)

train_losses = []  # courbe de loss train
val_losses   = []  # courbe de loss validation

print("Début de l'entraînement...")
for epoch in range(1, config.EPOCHS + 1):

    # -- Phase entraînement --
    model.train()
    total_loss, n = 0.0, 0
    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    avg_train_loss = total_loss / n
    train_losses.append(avg_train_loss)

    # -- Phase validation --
    model.eval()
    total_val_loss, n_val = 0.0, 0
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            total_val_loss += loss.item() * images.size(0)
            n_val += images.size(0)

    avg_val_loss = total_val_loss / n_val
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch:02d}/{config.EPOCHS} | Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")

# ---- Evaluation sur le jeu de test ----
print("Evaluation sur le jeu de test...")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_batch.numpy())

f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)
print(f"F1-score macro : {f1:.4f}")
print("Matrice de confusion :")
print(cm)

# ---- Courbe de loss train + validation ----
plt.figure()
plt.plot(range(1, config.EPOCHS + 1), train_losses, marker="o", label="Train")
plt.plot(range(1, config.EPOCHS + 1), val_losses,   marker="o", label="Validation")
plt.title("Courbe de loss - AlexNet")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_alexnet.png")
print("Courbe de loss sauvegardée dans loss_alexnet.png")

# ---- Sauvegarde du modèle ----
torch.save(model.state_dict(), "alexnet.pt")
print("Modèle sauvegardé dans alexnet.pt")