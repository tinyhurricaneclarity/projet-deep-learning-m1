# dataset.py

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

# Transformation des images : redimensionnement, conversion en tenseur, normalisation
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),  # redimensionne en 224x224
    transforms.ToTensor(),                                   # convertit en tenseur PyTorch
    transforms.Normalize(                                    # normalisation standard ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class RGBDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths  # liste des chemins vers les images
        self.labels = labels            # liste des labels (None pour val)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)   # nombre total d'images

    def __getitem__(self, i):
        # Charge l'image et applique les transformations
        img = Image.open(self.image_paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[i]
        return img

def load_data(root, train_dir):
    rgb_dir = os.path.join(root, train_dir, "RGB")
    image_paths = []
    labels = []

    for fname in sorted(os.listdir(rgb_dir)):
        if not fname.endswith(".png"):
            continue
        # Récupère la classe à partir du nom du fichier
        for lab in config.LABELS:
            if fname.startswith(lab):
                image_paths.append(os.path.join(rgb_dir, fname))
                labels.append(config.LBL2ID[lab])
                break

    return image_paths, labels

def split_data(image_paths, labels, test_frac=0.1, seed=config.SEED):
    # Fixe la graine aléatoire pour la reproductibilité
    torch.manual_seed(seed)

    # Sépare les indices par classe pour un split équilibré
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        class_indices[lab].append(i)

    test_idx, train_idx = [], []
    for lab, indices in class_indices.items():
        n_test = max(1, int(len(indices) * test_frac))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    # Crée les listes train et test
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_paths, train_labels, test_paths, test_labels