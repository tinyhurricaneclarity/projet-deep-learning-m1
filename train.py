# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# Création des datasets
train_dataset = RGBDataset(train_paths, train_labels, transform=transform)
test_dataset  = RGBDataset(test_paths,  test_labels,  transform=transform)

# Création des dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=config.NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

print(f"Train : {len(train_dataset)} images | Test : {len(test_dataset)} images")

# ---- Modèle AlexNet ----
print("Chargement d'AlexNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# Charge AlexNet pré-entraîné sur ImageNet
model = models.alexnet(pretrained=True)

# Remplace la dernière couche pour avoir 3 classes au lieu de 1000
model.classifier[6] = nn.Linear(4096, len(config.LABELS))
model = model.to(device)

# ---- Entraînement ----
criterion = nn.CrossEntropyLoss()  # fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)  # optimiseur

train_losses = []  # pour tracer la courbe de loss

print("Début de l'entraînement...")
for epoch in range(1, config.EPOCHS + 1):
    model.train()
    total_loss = 0
    n = 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()           # remet les gradients à zéro
        outputs = model(images)         # passe les images dans le modèle
        loss = criterion(outputs, labels_batch)  # calcule la perte
        loss.backward()                 # rétropropagation
        optimizer.step()               # mise à jour des poids

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    avg_loss = total_loss / n
    train_losses.append(avg_loss)
    print(f"Epoch {epoch:02d}/{config.EPOCHS} | Loss : {avg_loss:.4f}")

# ---- Evaluation ----
print("Evaluation sur le jeu de test...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_batch.numpy())

# Métriques
f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)
print(f"F1-score macro : {f1:.4f}")
print("Matrice de confusion :")
print(cm)

# ---- Courbe de loss ----
plt.figure()
plt.plot(range(1, config.EPOCHS + 1), train_losses, marker="o")
plt.title("Courbe de loss - AlexNet")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_alexnet.png")  # sauvegarde la courbe
print("Courbe de loss sauvegardée dans loss_alexnet.png")

# ---- Sauvegarde du modèle ----
torch.save(model.state_dict(), "alexnet.pt")
print("Modèle sauvegardé dans alexnet.pt")