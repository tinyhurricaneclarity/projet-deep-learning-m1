"""Evaluation sur test set et matrice de confusion sur meilleure configuration
Réentraînement avec meilleure configuration
"""

import model
import torch
import data_load
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from config import class_names, batch_size
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet50().to(device)
print(model)

path_train = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/MS/"

class_names = ["Health", "Rust", "Other"]
Im_type     = "RGB_MS"
Num_data    = 600  # 200 images par classe

# Chargement des données
split = torch.load("src/resnet50_RGB_MS/results/split_par_classe_RGB_MS_50.pth")

# Test set
test = data_load.import_images(class_names, split, "test")
test_dataset = data_load.CustomImageDataset(test['images'], test['labels'], transform=data_load.eval_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train set
train = data_load.import_images(class_names, split, "train")
train_dataset = data_load.CustomImageDataset(train['images'], train['labels'], transform=data_load.train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation set (CORRIGÉ)
val = data_load.import_images(class_names, split, "val")
val_dataset = data_load.CustomImageDataset(val['images'], val['labels'], transform=data_load.eval_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Hyperparamètres
num_epochs = 100
learning_rate = 0.0001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
step_size = 30
gamma = 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Listes pour stocker les résultats
train_losses, train_acc_list, val_losses, val_acc_list = [], [], [], []

# Sauvegarde du meilleur modèle
best_val_acc = 0.0
best_model_path = "results/best_resnet50_RGB_MS_model.pth"

print("Début de l'entraînement...")

for epoch in range(num_epochs):
    # ===== TRAIN =====
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

    # ===== VALIDATION =====
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
    
    scheduler.step(val_loss)
    
    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Meilleur modèle sauvegardé (epoch {epoch+1}, val_acc={val_acc:.2f}%)")
    
    # Affichage toutes les 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("\n" + "="*60)
print("Entraînement terminé !")
print(f"Meilleure validation accuracy : {best_val_acc:.2f}%")
print("="*60 + "\n")

# ===== ÉVALUATION FINALE SUR TEST SET =====
print("Chargement du meilleur modèle pour évaluation finale...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calcul des métriques
test_accuracy = accuracy_score(all_labels, all_preds) * 100
test_f1 = f1_score(all_labels, all_preds, average='weighted')
test_precision = precision_score(all_labels, all_preds, average='weighted')
test_recall = recall_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n" + "="*60)
print("RÉSULTATS SUR TEST SET")
print("="*60)
print(f"Accuracy  : {test_accuracy:.2f}%")
print(f"F1-Score  : {test_f1:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print("="*60 + "\n")

# ===== VISUALISATIONS =====

# 1. Courbes de loss
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
plt.plot(epochs, val_losses, label='Val Loss', marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - ResNet50 RGB MS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/loss_resnet50_RGB_MS.png", dpi=150, bbox_inches='tight')
print("✓ Courbe de loss sauvegardée dans loss_resnet50.png")

# 2. Courbes d'accuracy
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_acc_list, label='Train Accuracy', marker='o', markersize=3)
plt.plot(epochs, val_acc_list, label='Val Accuracy', marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Courbe d'accuracy - ResNet50 RGB_MS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/accuracy_resnet50_RGB_MS.png", dpi=150, bbox_inches='tight')
print("✓ Courbe d'accuracy sauvegardée dans accuracy_resnet50_RGB.png")

# 3. Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Matrice de confusion Resnet50 RGB MS - Test Set\nAccuracy: {test_accuracy:.2f}%')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.savefig("results/confusion_matrix_resnet50_RGB_MS.png", dpi=150, bbox_inches='tight')
print("✓ Matrice de confusion sauvegardée dans confusion_matrix_resnet50_RGB_MS.png")

print("\n✓ Toutes les visualisations ont été sauvegardées !")