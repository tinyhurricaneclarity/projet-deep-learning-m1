"""On fait des fonctions pour éviter d'avoir des variables gloables qui peuvent poser problèmes"""


import skimage as ski
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

#Définition des chemins pour tester les fonctions
path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
path_test_kaggle_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/RGB/" #pour tester sur des images non labellisé. Par vraiment un val


# Import des données : 600 images train

def load_data_train(path_train_rgb):

    x_train = [] #liste qui stocke les images
    y_train = [] #liste qui stocke les labels


    for i in range(1,201,1):
        path_health = f"{path_train_rgb}/Health_hyper_{i}.png"
        path_other = f"{path_train_rgb}/Other_hyper_{i}.png"
        path_rust = f"{path_train_rgb}/Rust_hyper_{i}.png"
        x_train.append(ski.io.imread(path_health))
        x_train.append(ski.io.imread(path_other))
        x_train.append(ski.io.imread(path_rust))
        y_train.append(0) #"health"
        y_train.append(1) #"other"
        y_train.append(2) #"rust"

    return x_train, y_train

#x_train, y_train = load_data_train(path_train_rgb, path_train_val)
#print("Nombre images train:", len(y_train))

# Import des images tests rgb + labels

def load_data_test_kaggle(path_test_kaggle_compet_rgb):
    x_test = []
    for rgb in sorted(Path(path_test_kaggle_compet_rgb).glob("*.png")): #prends toutes les images qui se terminent par rgb. glob ne marche que sur les objet Path
        x_test.append((ski.io.imread(rgb)))  # Ajoute l'image'
    
    return x_test

#x_test = load_data_test_kaggle(path_test_kaggle_rgb)
#print("Nombre images test:", len(x_test)) #300


#Conertion en Tensor 

to_tensor = transforms.ToTensor()

class CustomImageDataset(Dataset): #herite de Dataset de Pytorch
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): #l'index est géré automatiquement par PyTorch lorsque tu utilises un DataLoader.
        image = self.images[idx]  # récupère l’image à l'index correspondant
        label = self.labels[idx]
        
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long) #les labels sont des int, mais on convertit en tensor pour que ce soit compatible avec Pytorch
        #torch.long est un entier 64 bits signé (int64).

        return image, label #retourne un tuple (image_tensor, label).


#Chargement des datasets pour prendre des images de manière aléatoire dans le dataset

def create_dataloader(dataset, batch_size=32, seed=42):
    dataset_size = len(dataset)

    #split dataset : 70% training 10% validation et 20% test
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #on shuffle pour éviter d'apprendre l'ordre des données
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_size, val_size
    
