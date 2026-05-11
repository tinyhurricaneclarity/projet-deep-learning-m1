""" RESNET18 HS Train avec grid_search
import des data PAR CLASSE. Hasard dans les classes (Health, Rust, Other)
AVEC data augmentation

 """


#Import des packages

import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import skimage as ski

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import random

import data_load
import model as model_module


#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import des données train provenant de data.py

### CHOIX DES INPUTS ET PARAMETRES

class_names = ["Health", "Rust", "Other"]
Im_type     = "HS"
Num_data    = 600  # 200 images par classe

path = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"

# Fabrication du split aléatoire PAR CLASSE
# 99 val (33 par classe) + 99 test (33 par classe) + 402 train (134 par classe)
dico_train_test = data_load.alea_train_test(Num_data, class_names, n_val=99, n_test=99)
data_load.sufix_and_path(Im_type, dico_train_test, path)


# Chargement des images
Train = data_load.import_images(class_names, dico_train_test, "train")
Val   = data_load.import_images(class_names, dico_train_test, "val")
Test  = data_load.import_images(class_names, dico_train_test, "test")

print(f"Train : {len(Train['images'])} images")
print(f"Val   : {len(Val['images'])} images")
print(f"Test  : {len(Test['images'])} images")

# sauvegarde test loader indices (pour garder les memes indices pour l'évaluation dans eval.py)

os.makedirs("src/resnet18_HS/results", exist_ok=True)
torch.save(dico_train_test, "src/resnet18_HS/results/split_par_classe_data_augmentation_MS.pth") #dictionnaire {"images": np.array(images), "labels": labels}


#Convertion en tensor et trainloader


### DATASETS ET DATALOADERS

batch_size = 32

train_dataset = data_load.CustomImageDataset(Train['images'], Train['labels'], transform=data_load.train_transform)
val_dataset   = data_load.CustomImageDataset(Val['images'],   Val['labels'],   transform=data_load.eval_transform)
test_dataset  = data_load.CustomImageDataset(Test['images'],  Test['labels'],  transform=data_load.eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


#Fonction loss
criterion = nn.CrossEntropyLoss()

#Initialisation des variables pour stocker le meilleur modèle
best_val_loss = float('inf') #initialise à l'infini positif (donc un nbr positif tout simplement), parce que la loss est tjrs >0
best_val_acc = 0

os.makedirs("src/resnet18_HS/results/saved_models", exist_ok=True)

#Grid search des hyperparamètres

#Syntaxe pour futur grid search
grid_params = {
    "num_epochs": [50],
    "learning_rate": [0.001, 0.0001],
    "optimizer": ["Adam", "SGD"],
    "scheduler": ["StepLR", "ReduceLROnPlateau"],
    "step_size": [20, 30],
    "gamma": [0.1, 0.5],
    #"dropout": [0.0, 0.3, 0.5] #pas de dropout pour l'instant. Ajout selon l'overfitting

}

results_summary = []
config_counter = 0

for (num_epochs, learning_rate, optimizer_name, scheduler_name, step_size, gamma) in itertools.product(
    grid_params["num_epochs"],
    grid_params["learning_rate"],
    grid_params["optimizer"],
    grid_params["scheduler"],
    grid_params["step_size"],
    grid_params["gamma"],
    #grid_params["dropout"]
):

    # Create a dictionary to store the current configuration
    config = {
        # training
        "num_epochs": num_epochs,
        "batch_size":batch_size,

        # optimization
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "momentum": 0.9 if optimizer_name == "SGD" else None,

        # scheduler
        "scheduler": scheduler_name,
        "step_size": step_size if scheduler_name == "StepLR" else None,
        "gamma": gamma,

        #"dropout": dropout
    }

    # Generate a string to identify the current config
    config_str = (
    f"config{config_counter}_"
    f"epochs{num_epochs}_"
    f"lr{learning_rate}_"
    f"opt{optimizer_name}_"
    f"sched{scheduler_name}_"
    f"step{step_size}_"
    f"gamma{gamma}"
    )
    
    print(f"Running config {config_str}")
    config_counter += 1


    #Dans chaque boucle du grid search : Initialiser le modèle avec les hyperparamètres du grid search
    model = model_module.ResNet18().to(device) #lance une nouvelle instance du modèle, ca réinitialise tout.

    #Configuration des optimizer et scheduler du modèle qu'on vient d'initialiser

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),lr =learning_rate, momentum=0.9)
    
    #Configuration des scheduler

    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',        # réduit quand la val loss ne s'améliore plus
            factor=0.5,        # divise le learning rate par 2
            patience=5        # attend 5 epochs sans amélioration avant de réduire 
        )
    
    #Liste pour stocker les résultats (loss, acc)
    train_losses, train_acc_list, val_losses,  val_acc_list = [], [], [], []


    #entrainement du modèle

    for epoch in range(num_epochs):
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

    #évaluation des train et val

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
                correct += predicted.eq(labels).sum().item() #(y_pred == target)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_acc_list.append(val_acc)
        
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()


        
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

    #Sauvegarde du meilleur modèle sur base loss et acc (ancien modèle écrasé si meilleur)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "src/resnet18_MS/results/saved_models/best_loss_ms_data_aug.pth")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "src/resnet18_MS/results/saved_models/best_acc_ms_data_aug.pth")

    #Test pour chaque config
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images       = images.to(device)
            labels       = labels.squeeze() #nettoyage
            outputs      = model(images) #prédiction par le modèle sur les images du test set. sortie : [batch_size, nb_classes]. ex : [32, 3] → 3 classes
            _, predicted = torch.max(outputs, 1) #torch.max(input, dim). Dans la dimension 1 de outputs (donc les classes), on prend la classe avec la plus grande probabilité (torch.max retourne deux valeurs : (max_value, max_index))
            all_preds.extend(predicted.cpu().numpy()) #stockage des prédictions sous forme numpy. .extend permet d'ajouter plusieurs éléments à une liste
            all_labels.extend(labels.numpy()) #stockage des vrais labels

    #ajout des résultats dans un dictionnaire
    f1_par_classe = f1_score(all_labels, all_preds, average=None)
    results_summary.append({
        
        "config": config_str,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_Health":   f1_par_classe[0],
        "f1_Rust":     f1_par_classe[1],
        "f1_Other":    f1_par_classe[2]
    })


#Résumé des résultats dans un ficher CSV
df = pd.DataFrame(results_summary)
df.to_csv("src/resnet18_HS/results/grid_search_results_par_classe_data_augmentation_HS.csv", index=False)
print("CSV résultats.")

# Affichage meilleur modèle
print("\n─── Résumé grid search ───")
for r in sorted(results_summary, key=lambda x: x["val_acc"], reverse=True)[:5]:
    print(f"  {r['config']}  →  val_acc={r['val_acc']:.2f}%  val_loss={r['val_loss']:.4f}")

