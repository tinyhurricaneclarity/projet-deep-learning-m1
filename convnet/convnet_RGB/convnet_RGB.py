# convnet_RGB.py
# ConvNet from scratch sur images RGB 64x64

# Objectif : entraîner un réseau de neurones convolutif (ConvNet) from scratch sur les images RGB du dataset Kaggle "Beyond Visible Spectrum".
# Le modèle doit classifier les images en 3 classes : Health, Rust, Other.

# Améliorations apportées :
# - Data augmentation dynamique (flips, rotation) pour réduire l'overfitting
# - Tirage au sort à l'intérieur de chaque classe pour l'équilibre dans chaque sous-ensemble (train, val, test)
# - Val : 99 images (33 par classe), Test : 99 images (33 par classe)
# - 100 epochs pour laisser le temps au modèle d'apprendre
# - Learning rate scheduler qui réduit automatiquement le learning rate quand la val loss stagne
# - Sauvegarde du meilleur modèle basée sur la val loss
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
    """
    Convertit un pourcentage en proportion.
    Ex : 70 -> 0.7 / 0.7 -> 0.7 (déjà une proportion)
    """
    while (pourcent > 100) or (pourcent < 0):
        print("La valeur de proportion ou pourcentage est hors des clous !")
        return "crash"
    return pourcent if pourcent < 1 else pourcent / 100


def alea_train_test(Num_data, class_names, n_val=99, n_test=99):
    """
    Crée un split aléatoire des données en train / validation / test
    EN TIRANT AU SORT A L'INTERIEUR DE CHAQUE CLASSE.
    
    Cela garantit que chaque classe est représentée équitablement
    dans chaque sous-ensemble — conformément aux recommandations de la tutrice.
    
    Num_data : nombre total d'images (toutes classes confondues)
    class_names : liste des noms de classes
    n_val : nombre total d'images pour la validation (99 = 33 par classe)
    n_test : nombre total d'images pour le test (99 = 33 par classe)
    """
    n_per_class      = int(Num_data / len(class_names))  # 200 images par classe
    n_val_per_class  = int(n_val / len(class_names))     # 33 images par classe en val
    n_test_per_class = int(n_test / len(class_names))    # 33 images par classe en test

    train_indices = []
    val_indices   = []
    test_indices  = []

    for j in range(len(class_names)):
        # tirage au sort des indices POUR CETTE CLASSE uniquement
        indices = random.sample(range(1, n_per_class + 1), n_per_class)

        # val : les 33 premiers indices tirés au sort
        val_indices.extend([(j, indices[i]) for i in range(n_val_per_class)])

        # test : les 33 suivants
        test_indices.extend([(j, indices[i]) for i in range(n_val_per_class,
                                                             n_val_per_class + n_test_per_class)])

        # train : tout le reste (200 - 33 - 33 = 134 par classe)
        train_indices.extend([(j, indices[i]) for i in range(n_val_per_class + n_test_per_class,
                                                              n_per_class)])

    return {
        "train": train_indices,
        "val":   val_indices,
        "test":  test_indices
    }


def sufix_and_path(Im_type, dico, path):
    """
    Ajoute au dictionnaire le suffixe et le chemin correspondant au type d'image.
    RGB -> .png / MS et HS -> .tif
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


def import_images(class_names, dico, tr_or_te="train"):
    """
    Charge les images depuis le disque pour le split demandé (train, val ou test).
    Compatible avec le tirage au sort par classe.
    
    Les indices sont des tuples (j, i) où :
    - j = indice de la classe (0=Health, 1=Rust, 2=Other)
    - i = numéro de l'image dans cette classe
    """
    images = []
    labels = []

    indices = dico[tr_or_te]

    for (j, i) in indices:
        tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
        images.append(ski.io.imread(tpr_path))
        labels.append(j)

    return {"images": np.array(images), "labels": labels}



### TRANSFORMATIONS DES IMAGES

# Pour le train : avec data augmentation dynamique
# A chaque epoch, une transformation aléatoire différente est appliquée
# ce qui force le modèle à généraliser plutôt qu'apprendre par coeur
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),   # flip horizontal aléatoire
    transforms.RandomVerticalFlip(),     # flip vertical aléatoire
    transforms.RandomRotation(15),       # rotation aléatoire jusqu'à 15 degrés
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Pour val et test : sans augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



### CLASSE DATASET

class CustomImageDataset(Dataset):
    """
    Classe Dataset personnalisée compatible avec PyTorch.
    Doit implémenter __len__ et __getitem__.
    """
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
    ConvNet from scratch adapté aux images RGB 64x64.
    3 blocs convolutifs + couches fully connected.
    """
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()

        # Bloc 1 : détecte des features simples (bords, textures)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Bloc 2 : détecte des features plus complexes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Bloc 3 : détecte des features encore plus abstraites
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # MaxPool : réduit la taille spatiale par 2
        self.pool    = nn.MaxPool2d(2, 2)
        # Dropout : désactive 30% des neurones pour éviter l'overfitting
        self.dropout = nn.Dropout(0.3)
        self.relu    = nn.ReLU()

        # après 3 maxpoolings sur 64x64 : 64/2/2/2 = 8
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



### CHOIX DES INPUTS ET PARAMETRES

class_names = ["Health", "Rust", "Other"]
Im_type     = "RGB"
Num_data    = 600  # 200 images par classe

path = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"

# Fabrication du split aléatoire PAR CLASSE
# 99 val (33 par classe) + 99 test (33 par classe) + 402 train (134 par classe)
random.seed(3557)
dico_train_test = alea_train_test(Num_data, class_names, n_val=99, n_test=99)
sufix_and_path(Im_type, dico_train_test, path)

# Chargement des images
Train = import_images(class_names, dico_train_test, "train")
Val   = import_images(class_names, dico_train_test, "val")
Test  = import_images(class_names, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")




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
    mode='min',        # réduit quand la val loss ne s'améliore plus
    factor=0.5,        # divise le learning rate par 2
    patience=5,        # attend 5 epochs sans amélioration avant de réduire
)



### BOUCLE D'ENTRAINEMENT

train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_val_loss            = float('inf')
epochs_sans_amelioration = 0
patience                 = 15  # arrête si pas d'amélioration pendant 15 epochs
stop_training            = False 

os.makedirs("saved_models", exist_ok=True)

print("Début de l'entraînement.")
for epoch in range(1, num_epochs + 1):

    if stop_training:
        continue  # saute les epochs restantes

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

    # Sauvegarde du meilleur modèle + early stopping
    if val_loss < best_val_loss:
        best_val_loss            = val_loss
        epochs_sans_amelioration = 0
        torch.save(model.state_dict(), "saved_models/convnet_RGB_best.pth")
        print(f"  -> Meilleur modèle sauvegardé à l'epoch {epoch} (Val Loss: {val_loss:.4f})")
    else:
        epochs_sans_amelioration += 1
        if epochs_sans_amelioration >= patience:
            print(f"Early stopping déclenché à l'epoch {epoch} — pas d'amélioration depuis {patience} epochs.")
            stop_training = True



### EVALUATION FINALE SUR LE JEU DE TEST

print("\nChargement du meilleur modèle.")    # on recharge le meilleur modèle sauvegardé
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

# F1-score global
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"F1-score macro : {f1:.4f}")
# F1 score macro calcule la moyenne du F1 score sur les 3 classes de manière équilibrée

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print("Matrice de confusion :")
print(cm)

# F1-score par classe
f1_par_classe = f1_score(all_labels, all_preds, average=None)
for i, classe in enumerate(class_names):
    print(f"F1-score {classe} : {f1_par_classe[i]:.4f}")



### COURBES D'APPRENTISSAGE


epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses,   label='Val Loss',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de loss - ConvNet RGB")
plt.legend()
plt.savefig("loss_convnet_RGB.png")
print("Courbe de loss sauvegardée dans loss_convnet_RGB.png")

plt.figure(figsize=(10, 4))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accuracies,   label='Val Accuracy',   marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Courbe d'accuracy - ConvNet RGB")
plt.legend()
plt.savefig("accuracy_convnet_RGB.png")
print("Courbe d'accuracy sauvegardée dans accuracy_convnet_RGB.png")

print("\nTerminé.")
