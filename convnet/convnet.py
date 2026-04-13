# convnet.py
# ConvNet from scratch sur images RGB
# Basé sur la structure de code de Léa 
# Améliorations : data augmentation + sauvegarde meilleur modèle

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

# --- FONCTIONS (Léa et Maël) ---

def pourcent_to_prop(pourcent):
    """
    Pour un nombre supérieur à 1 et inférieur ou égal à 100, retourne la proportion.
    Si c'est entre 0 et 1, retourne la valeur d'origine.
    """
    while (pourcent > 100) or (pourcent < 0):
        print("La valeur de proportion ou pourcentage est hors des clous !")
        return "crash"
    return pourcent if pourcent < 1 else pourcent / 100


def alea_train_test(Num_data, class_names):
    """
    Retourne un dictionnaire contenant la liste des indices des images
    dans un ordre aléatoire et les ranges des parties train, validation et test.
    """
    tpr = int(Num_data / len(class_names))
    alea_liste = random.sample(range(1, tpr + 1), tpr)
    ind_prop_train = int(len(alea_liste) * proportion_train)
    ind_prop_val   = int(len(alea_liste) * proportion_validation)
    ind_prop_test  = int(len(alea_liste) * proportion_test)

    prop_liste_train = range(0, ind_prop_train)
    prop_liste_val   = range(ind_prop_train, ind_prop_train + ind_prop_val)
    prop_liste_test  = range(ind_prop_train + ind_prop_val, ind_prop_train + ind_prop_val + ind_prop_test)

    return {
        "alea": alea_liste,
        "prop_liste_train": prop_liste_train,
        "prop_liste_val":   prop_liste_val,
        "prop_liste_test":  prop_liste_test
    }


def sufix_and_path(Im_type, dico, path):
    """
    Ajoute à un dictionnaire existant le suffixe et chemin associé au type d'image.
    """
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


def import_images(class_names, dico, tr_or_te="none"):
    """
    Importe les images de train, val ou test à partir du dictionnaire configuré.
    """
    images = []
    labels = []

    if tr_or_te == "train":
        key = "prop_liste_train"
    elif tr_or_te == "val":
        key = "prop_liste_val"
    elif tr_or_te == "test":
        key = "prop_liste_test"
    else:
        key = "none"

    for k in dico[key]:
        i = dico["alea"][k]
        for j in range(len(class_names)):
            tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
            images.append(ski.io.imread(tpr_path))
            labels.append(j)

    return {"images": np.array(images), "labels": labels}


# --- TRANSFORMATIONS ---

# Pour le train : avec data augmentation
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),    # flip horizontal aléatoire
    transforms.RandomVerticalFlip(),      # flip vertical aléatoire
    transforms.RandomRotation(15),        # rotation aléatoire jusqu'à 15 degrés
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Pour val et test : sans augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- DATASET ---

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


# --- ARCHITECTURE CONVNET ---

class ConvNet(nn.Module):
    """
    ConvNet from scratch adapté aux images RGB 64x64.
    3 blocs convolutifs + couches fully connected.
    """
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()

        # Bloc 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Bloc 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Bloc 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)  # divise la taille par 2
        self.dropout = nn.Dropout(0.3)     # evite l'overfitting
        self.relu    = nn.ReLU()

        # après 3 poolings sur 64x64 : 64/2/2/2 = 8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> 32x32
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> 16x16
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # -> 8x8
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- CHOIX DES INPUTS ---

class_names           = ["Health", "Rust", "Other"]
Im_type               = "RGB"
Num_data              = 600
proportion_train      = 70
proportion_validation = 10
proportion_test       = 20

path = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"

# Conversion pourcentages -> proportions
proportion_train      = pourcent_to_prop(proportion_train)
proportion_validation = pourcent_to_prop(proportion_validation)
proportion_test       = pourcent_to_prop(proportion_test)

# Fabrication du split aléatoire
dico_train_test = alea_train_test(Num_data, class_names)
sufix_and_path(Im_type, dico_train_test, path)

# Import des images
Train = import_images(class_names, dico_train_test, "train")
Val   = import_images(class_names, dico_train_test, "val")
Test  = import_images(class_names, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")

# --- HISTOGRAMME des classes ---
counts = Counter(Train['labels'])
values = [counts[0], counts[1], counts[2]]
plt.figure(figsize=(6, 4))
plt.bar(class_names, values)
plt.title("Distribution des images par classe (Train)")
plt.xlabel("Classes")
plt.ylabel("Nombre d'images")
plt.savefig("histogramme_classes.png")
print("Histogramme sauvegardé dans histogramme_classes.png")

# --- DATASETS et DATALOADERS ---
batch_size    = 32
num_epochs    = 50
learning_rate = 0.001

train_dataset = CustomImageDataset(Train['images'], Train['labels'], transform=train_transform)
val_dataset   = CustomImageDataset(Val['images'],   Val['labels'],   transform=eval_transform)
test_dataset  = CustomImageDataset(Test['images'],  Test['labels'],  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# --- MODELE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

model     = ConvNet(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- ENTRAINEMENT ---
train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_val_loss = float('inf')  # pour sauvegarder le meilleur modèle
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

    # Sauvegarde du meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "saved_models/convnet_best.pth")
        print(f"  -> Meilleur modèle sauvegardé à l'epoch {epoch} (Val Loss: {val_loss:.4f})")

# --- EVALUATION SUR LE JEU DE TEST ---
# On charge le meilleur modèle sauvegardé
print("\nChargement du meilleur modèle")
model.load_state_dict(torch.load("saved_models/convnet_best.pth"))

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
print(f"F1-score macro : {f1:.4f}")
print("Matrice de confusion :")
print(cm)

# --- COURBES ---
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses,   label='Val Loss',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - ConvNet")
plt.legend()
plt.savefig("loss_convnet.png")
print("Courbe de loss sauvegardée dans loss_convnet.png")

plt.figure(figsize=(10, 4))
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracies,   label='Val Accuracy',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Courbe d'accuracy - ConvNet")
plt.legend()
plt.savefig("accuracy_convnet.png")
print("Courbe d'accuracy sauvegardée dans accuracy_convnet.png")

print("\nTerminé.")
