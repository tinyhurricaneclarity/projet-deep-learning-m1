# convnet_HS.py
# ConvNet from scratch sur images HS 32x32

# Objectif : entraîner un réseau de neurones convolutif (ConvNet) from scratch sur les images HS du dataset.
# Le modèle doit classifier les images en 3 classes : Health, Rust, Other

# Améliorations apportées :
# - Data augmentation dynamique (flips, rotation)
# - Tirage au sort à l'intérieur de chaque classe pour l'équilibre
# - Val : 99 images (33 par classe), Test : 99 images (33 par classe)
# - 100 epochs
# - Learning rate scheduler (ReduceLROnPlateau)
# - Sauvegarde automatique du meilleur modèle basée sur la val loss
# - F1-score par classe pour identifier quelle classe sous-performe


### IMPORTS

import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix


### FONCTIONS (de Léa et Maël)

def pourcent_to_prop(pourcent):
    while (pourcent > 100) or (pourcent < 0):
        print("La valeur de proportion ou pourcentage est hors des clous !")
        return "crash"
    return pourcent if pourcent < 1 else pourcent / 100


def alea_train_test(Num_data, class_names, n_val=99, n_test=99):
    n_per_class      = int(Num_data / len(class_names))
    n_val_per_class  = int(n_val / len(class_names))
    n_test_per_class = int(n_test / len(class_names))

    train_indices = []
    val_indices   = []
    test_indices  = []

    for j in range(len(class_names)):
        indices = random.sample(range(1, n_per_class + 1), n_per_class)
        val_indices.extend([(j, indices[i]) for i in range(n_val_per_class)])
        test_indices.extend([(j, indices[i]) for i in range(n_val_per_class,
                                                             n_val_per_class + n_test_per_class)])
        train_indices.extend([(j, indices[i]) for i in range(n_val_per_class + n_test_per_class,
                                                              n_per_class)])
    return {
        "train": train_indices,
        "val":   val_indices,
        "test":  test_indices
    }


def sufix_and_path(Im_type, dico, path):
    if Im_type == "RGB":
        dico["sufix"]     = ".png"
        dico["path_data"] = f"{path}/RGB"
    else:
        dico["sufix"] = ".tif"
        if Im_type == "MS":
            dico["path_data"] = f"{path}/MS"
        elif Im_type == "HS":
            dico["path_data"] = f"{path}/HS"
        else:
            print("Forme inconnue")
            

def import_images(class_names, dico, tr_or_te="train"):
    images = []
    labels = []
    indices = dico[tr_or_te]
    for (j, i) in indices:
        tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
        img = ski.io.imread(tpr_path)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Force 125 canaux : certaines images HS peuvent avoir un nombre différent
        if img.shape[2] < 125:
            # padding avec des zéros si moins de 125 canaux
            pad = np.zeros((32, 32, 125 - img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
        elif img.shape[2] > 125:
            # coupe si plus de 125 canaux
            img = img[:, :, :125]
        
        images.append(img)
        labels.append(j)
    return {"images": np.array(images), "labels": labels}


### TRANSFORMATIONS DES IMAGES

# Les images HS ont 125 canaux
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.5] * 125, std=[0.5] * 125)
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 125, std=[0.5] * 125)
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


### ARCHITECTURE CONVNET

class ConvNet(nn.Module):
    """
    ConvNet from scratch adapté aux images HS 32x32.
    Les images HS ont 125 canaux et une taille de 32x32 pixels.
    3 blocs convolutifs + couches fully connected.
    """
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()

        # Bloc 1 : 125 canaux en entrée (HS)
        self.conv1 = nn.Conv2d(125, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Bloc 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Bloc 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu    = nn.ReLU()

        # après 3 maxpoolings sur 32x32 : 32/2/2/2 = 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> 16x16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> 8x8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # -> 4x4
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


### CHOIX DES INPUTS ET PARAMETRES

class_names = ["Health", "Rust", "Other"]
Im_type     = "HS"
Num_data    = 600

path = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"

dico_train_test = alea_train_test(Num_data, class_names, n_val=99, n_test=99)
sufix_and_path(Im_type, dico_train_test, path)

Train = import_images(class_names, dico_train_test, "train")
Val   = import_images(class_names, dico_train_test, "val")
Test  = import_images(class_names, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")


### HISTOGRAMMES DES CLASSES

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title, color in zip(
    axes,
    [Train, Val, Test],
    ["Train", "Val", "Test"],
    ["blue", "orange", "green"]
):
    counts = Counter(data['labels'])
    values = [counts[0], counts[1], counts[2]]
    ax.bar(class_names, values, color=color)
    ax.set_title(f"Distribution ({title})")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Nombre d'images")

plt.tight_layout()
plt.savefig("histogrammes_complets_HS.png")
print("Histogrammes sauvegardés dans histogrammes_complets_HS.png")


### HYPERPARAMETRES

batch_size    = 32
num_epochs    = 100
learning_rate = 0.001


### DATASETS ET DATALOADERS

train_dataset = CustomImageDataset(Train['images'], Train['labels'], transform=train_transform)
val_dataset   = CustomImageDataset(Val['images'],   Val['labels'],   transform=eval_transform)
test_dataset  = CustomImageDataset(Test['images'],  Test['labels'],  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


### INITIALISATION DU MODELE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

model     = ConvNet(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
)


### BOUCLE D'ENTRAINEMENT

train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_val_loss = float('inf')
os.makedirs("saved_models", exist_ok=True)

print("Début de l'entraînement.")
for epoch in range(1, num_epochs + 1):

    # Phase train
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

    # Phase validation
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

    print(f"Epoch [{epoch:02d}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "saved_models/convnet_HS_best.pth")
        print(f"  -> Meilleur modèle sauvegardé à l'epoch {epoch} (Val Loss: {val_loss:.4f})")


### EVALUATION FINALE SUR LE JEU DE TEST

print("\nChargement du meilleur modèle.")
model.load_state_dict(torch.load("saved_models/convnet_HS_best.pth"))

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
print(f"F1-score macro : {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("Matrice de confusion :")
print(cm)

f1_par_classe = f1_score(all_labels, all_preds, average=None)
for i, classe in enumerate(class_names):
    print(f"F1-score {classe} : {f1_par_classe[i]:.4f}")


### COURBES D'APPRENTISSAGE

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses,   label='Val Loss',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - ConvNet HS")
plt.legend()
plt.savefig("loss_convnet_HS.png")
print("Courbe de loss sauvegardée dans loss_convnet_HS.png")

plt.figure(figsize=(10, 4))
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracies,   label='Val Accuracy',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Courbe d'accuracy - ConvNet HS")
plt.legend()
plt.savefig("accuracy_convnet_HS.png")
print("Courbe d'accuracy sauvegardée dans accuracy_convnet_HS.png")

print("\nTerminé.")