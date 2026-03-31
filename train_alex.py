"""
Détection de Maladies de Plantes - AlexNet ne pas utiliser que squelette
3 classes : 
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


#   À MODIFIER

TRAIN_DIR  = "/net/cremi/llorella/Bureau/train"   # dossier contenant les .png

EPOCHS     = 25
BATCH_SIZE = 32
LR         = 0.001


CLASSES = ["Healthy", "Other", "Rust"]


# Dataset personnalisé : lit le label depuis le nom du fichier
class PlantDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.samples   = []

        for filename in os.listdir(folder):
            if not filename.endswith(".png"):
                continue
            # Cherche quelle classe correspond au début du nom de fichier
            label = None
            for i, cls in enumerate(CLASSES):
                if filename.startswith(cls):
                    label = i
                    break
            if label is None:
                continue  # fichier ignoré si aucune classe reconnue
            self.samples.append((os.path.join(folder, filename), label))

        print(f"  {len(self.samples)} images chargées dans {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Chargement des données
print("Chargement des données...")
train_ds     = PlantDataset(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Modèle AlexNet pré-entraîné
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Geler les couches convolutives
for param in model.features.parameters():
    param.requires_grad = False

# Adapter la sortie : 1000 → 3 classes
model.classifier[6] = nn.Linear(4096, len(CLASSES))
model = model.to(device)

# Loss et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# Entraînement

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_ds)
    loss = total_loss / len(train_ds)

    print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss={loss:.4f}  acc={acc:.4f}")

# Sauvegarde
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/alexnet_best.pth")
with open("models/class_names.json", "w") as f:
    json.dump(CLASSES, f)

print("\n Terminé.")
print("   Modèle  → models/alexnet_best.pth")
print("   Classes → models/class_names.json")
