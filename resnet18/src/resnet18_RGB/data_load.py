"""On fait des fonctions pour éviter d'avoir des variables gloables qui peuvent poser problèmes"""

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset


#Définition des chemins pour tester les fonctions
path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
path_test_kaggle_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/RGB/" #pour tester sur des images non labellisé. Par vraiment un val


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

x_train, y_train = load_data_train(path_train_rgb)
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

    #Création des loaders après le split 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #on shuffle pour éviter d'apprendre l'ordre des données
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_size, val_size
    

"""-------------------------------Pour importer par classe-------------------------------------------------"""



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



"""---------------------------------Data augmentation-------------------------------------"""

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
                         std=[0.229, 0.224, 0.225]) #permet de "lisser" la distribution des tensors (valeurs autour de 0) pour accélérer l’entraînement et rendre l’optimisation plus stable
])


# Pour val et test : sans augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])




