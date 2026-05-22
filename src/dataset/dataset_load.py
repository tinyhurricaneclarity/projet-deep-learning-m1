# src/dataset/dataset_load.py
"""
Chargement et préparation des données.
Adapté à toutes les modalités : RGB, MS, HS et fusions des modalités.
"""

import numpy as np
import skimage as ski
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.config.config import SEED_RANDOM, SEED_TORCH, CLASS_NAMES, BATCH_SIZE


### SEEDS

random.seed(SEED_RANDOM)
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)


### FONCTIONS DE SPLIT

def pourcent_to_prop(pourcent):
    """Convertit un pourcentage en proportion."""
    while (pourcent > 100) or (pourcent < 0):
        print("La valeur de proportion ou pourcentage est hors des clous !")
        return "crash"
    return pourcent if pourcent < 1 else pourcent / 100


def alea_train_test(Num_data, class_names, n_val=99, n_test=99):
    """
    Split aléatoire PAR CLASSE pour garantir l'équilibre entre les classes.
    
    Paramètres :
    - Num_data : nombre total d'images (ex: 600 pour 200 par classe)
    - class_names : liste des noms de classes
    - n_val : nombre total d'images pour validation (réparti équitablement)
    - n_test : nombre total d'images pour test (réparti équitablement)
    
    Retour : dictionnaire {"train": [...], "val": [...], "test": [...]}
    """
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
    """
    Ajoute le suffixe et le chemin selon le type d'image.
    
    Paramètres :
    - Im_type : type de modalité ('RGB', 'MS', 'HS', 'RGB_MS', etc.)
    - dico : dictionnaire contenant les indices de split
    - path : chemin racine vers les données
    """
    dico["path_data_root"] = path  # toujours défini
    dico["fusion"] = None          # par défaut pas de fusion

    if Im_type == "RGB":
        dico["sufix"]     = ".png"
        dico["path_data"] = f"{path}/RGB"
    elif Im_type == "MS" or Im_type == "MS_sans_other":
        dico["sufix"]     = ".tif"
        dico["path_data"] = f"{path}/MS"
    elif Im_type == "HS" or Im_type == "HS_sans_other":
        dico["sufix"]     = ".tif"
        dico["path_data"] = f"{path}/HS"
    elif Im_type in ("RGB_MS", "MS_HS", "RGB_MS_HS"):
        dico["fusion"] = Im_type
    else:
        print(f"Modalité inconnue : {Im_type}")


def load_single_image(path, is_hs=False):
    """
    Charge une seule image et la normalise entre 0 et 1.
    
    Paramètres :
    - path : chemin vers l'image
    - is_hs : si True, assure que l'image HS a exactement 125 canaux
    
    Retour : array numpy (H, W, C) normalisé
    """
    img = ski.io.imread(path).astype(np.float32)
    
    # Normalisation entre 0 et 1
    if img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    # Traitement spécifique pour HS (assurer 125 canaux)
    if is_hs and len(img.shape) == 3:
        if img.shape[2] < 125:
            # Padding si moins de 125 canaux
            pad = np.zeros((img.shape[0], img.shape[1], 125 - img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
        elif img.shape[2] > 125:
            # Troncature si plus de 125 canaux
            img = img[:, :, :125]
    
    return img


def import_images(class_names, dico, tr_or_te="train"):
    """
    Importe les images selon la modalité (simple ou fusion).
    
    Paramètres :
    - class_names : liste des noms de classes
    - dico : dictionnaire contenant les chemins et indices
    - tr_or_te : 'train', 'val' ou 'test'
    
    Retour : dictionnaire {"images": np.array, "labels": list}
    """
    images  = []
    labels  = []
    indices = dico[tr_or_te]
    path    = dico["path_data_root"]
    fusion  = dico.get("fusion", None)

    for (j, i) in indices:
        if fusion is None:
            # Modalité simple (RGB, MS, HS, MS_sans_other, HS_sans_other)
            tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
            is_hs = dico.get("path_data", "").endswith("HS")
            img = load_single_image(tpr_path, is_hs=is_hs)

        elif fusion == "RGB_MS":
            # Fusion RGB + MS
            rgb = load_single_image(f"{path}/RGB/{class_names[j]}_hyper_{i}.png")
            ms  = load_single_image(f"{path}/MS/{class_names[j]}_hyper_{i}.tif")
            img = np.concatenate([rgb, ms], axis=2)  # (64, 64, 8)

        elif fusion == "MS_HS":
            # Fusion MS + HS
            ms = load_single_image(f"{path}/MS/{class_names[j]}_hyper_{i}.tif")
            hs = load_single_image(f"{path}/HS/{class_names[j]}_hyper_{i}.tif", is_hs=True)
            # Resize HS de 32x32 à 64x64
            hs_resized = np.stack([
                ski.transform.resize(hs[:,:,k], (64,64), anti_aliasing=True)
                for k in range(hs.shape[2])
            ], axis=2).astype(np.float32)
            img = np.concatenate([ms, hs_resized], axis=2)  # (64, 64, 130)

        elif fusion == "RGB_MS_HS":
            # Fusion RGB + MS + HS
            rgb = load_single_image(f"{path}/RGB/{class_names[j]}_hyper_{i}.png")
            ms  = load_single_image(f"{path}/MS/{class_names[j]}_hyper_{i}.tif")
            hs  = load_single_image(f"{path}/HS/{class_names[j]}_hyper_{i}.tif", is_hs=True)
            # Resize HS de 32x32 à 64x64
            hs_resized = np.stack([
                ski.transform.resize(hs[:,:,k], (64,64), anti_aliasing=True)
                for k in range(hs.shape[2])
            ], axis=2).astype(np.float32)
            img = np.concatenate([rgb, ms, hs_resized], axis=2)  # (64, 64, 133)

        images.append(img)
        labels.append(j)

    return {"images": np.array(images), "labels": labels}


### TRANSFORMATIONS

def get_transforms(in_channels):
    """
    Retourne les transformations pour train et eval selon le nombre de canaux.
    
    RGB : normalisation ImageNet
    MS/HS : normalisation 0.5
    
    Paramètres :
    - in_channels : nombre de canaux d'entrée
    
    Retour : (train_transform, eval_transform)
    """
    if in_channels == 3:
        # Normalisation ImageNet pour RGB
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        # Normalisation générique pour MS/HS
        mean = [0.5] * in_channels
        std  = [0.5] * in_channels

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


### CLASSE DATASET

class CustomImageDataset(Dataset):
    """Dataset PyTorch pour nos images."""
    
    def __init__(self, images, labels, transform=None):
        """
        Paramètres :
        - images : numpy array de shape (N, H, W, C)
        - labels : liste de labels
        - transform : transformations à appliquer
        """
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
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label


### CRÉATION DES DATALOADERS

def create_dataloaders(Train, Val, Test, train_transform, eval_transform, batch_size=None):
    """
    Crée les DataLoaders pour train, val et test.
    
    Paramètres :
    - Train, Val, Test : dictionnaires {"images": array, "labels": list}
    - train_transform : transformations pour train
    - eval_transform : transformations pour val/test
    - batch_size : taille du batch (None = utilise BATCH_SIZE de config)
    
    Retour : (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    train_dataset = CustomImageDataset(Train['images'], Train['labels'], transform=train_transform)
    val_dataset   = CustomImageDataset(Val['images'],   Val['labels'],   transform=eval_transform)
    test_dataset  = CustomImageDataset(Test['images'],  Test['labels'],  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader