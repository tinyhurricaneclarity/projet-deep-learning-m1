"""
Prédiction - Détection de Maladies de Plantes
Utilisation : python predict.py chemin/vers/image.png
"""

import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = "models/alexnet_best.pth"
CLASSES_PATH = "models/class_names.json"

# Vérification argument
if len(sys.argv) < 2:
    print("Usage : python predict.py chemin/vers/image.png")
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Erreur : image introuvable -> {image_path}")
    sys.exit()

# Chargement des classes
with open(CLASSES_PATH) as f:
    class_names = json.load(f)

# Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(weights=None)
model.classifier[6] = nn.Linear(4096, len(class_names))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transformations (identiques à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# Chargement image
image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)

# Prédiction
with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze()

# Affichage résultats
print(f"\nImage : {image_path}\n")

for i, cls in enumerate(class_names):
    barre = "█" * int(probs[i].item() * 30)
    print(f"{cls:10s} {barre:30s} {probs[i].item()*100:.1f}%")

predicted = class_names[probs.argmax().item()]
confidence = probs.max().item() * 100

print(f"\n→ Classe prédite : {predicted} ({confidence:.1f}% de confiance)")
