import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd
from pathlib import Path
import skimage as ski

from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

##Fonctions d'import des données


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


##Téléchargement des données

class_names = ["Health", "Rust", "Other"]
Im_type     = "RGB"
Num_data    = 600  # 200 images par classe

path = "/autofs/unityaccount/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"


# Fabrication du split aléatoire PAR CLASSE
# 99 val (33 par classe) + 99 test (33 par classe) + 402 train (134 par classe)
dico_train_test = alea_train_test(Num_data, class_names, n_val=99, n_test=99)
sufix_and_path(Im_type, dico_train_test, path)

# Chargement des images
Train = import_images(class_names, dico_train_test, "train")
Val   = import_images(class_names, dico_train_test, "val")
Test  = import_images(class_names, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")
