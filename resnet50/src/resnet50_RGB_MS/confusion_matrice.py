"""Script pour regénérer uniquement la matrice de confusion avec une meilleure police"""

import model
import torch
import data_load
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from config import class_names, batch_size
import matplotlib.pyplot as plt
import seaborn as sns

# Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.ResNet50().to(device)

class_names = ["Health", "Rust", "Other"]
best_model_path = "results/best_resnet50_RGB_MS_model.pth"

# Chargement des données de test
split = torch.load("src/resnet50_RGB_MS/results/split_par_classe_RGB_MS_50.pth")
test = data_load.import_images(class_names, split, "test")
test_dataset = data_load.CustomImageDataset(test['images'], test['labels'], transform=data_load.eval_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Chargement du modèle entraîné
print("Chargement du meilleur modèle...")
net.load_state_dict(torch.load(best_model_path, map_location=device))
net.eval()

# Prédictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calcul des métriques
test_accuracy = accuracy_score(all_labels, all_preds) * 100
conf_matrix = confusion_matrix(all_labels, all_preds)

# Matrice de confusion avec police agrandie
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 18})  # Taille des chiffres

plt.title(f'Matrice de confusion Resnet 50 RGB MS - Test Set\nAccuracy: {test_accuracy:.2f}%', 
          fontsize=18)
plt.ylabel('Vraie classe', fontsize=16)
plt.xlabel('Classe prédite', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("results/confusion_matrix_resnet50_RGB_MS_new.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Matrice de confusion régénérée avec accuracy: {test_accuracy:.2f}%")