"""Train avec grid_search
import des data au hasard """


#Import des packages

import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import skimage as ski

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import data_load
import model as model_module


#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Import des données train provenant de data.py

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
x_train, y_train = data_load.load_data_train(path_train_rgb)

#Convertion en tensor et trainloader
dataset = data_load.CustomImageDataset(x_train, y_train, transform=None)

#Dataloader
batch_size = 32
train_loader, val_loader, test_loader, _, _ = data_load.create_dataloader(dataset, batch_size=batch_size)

# sauvegarde test loader indices (pour garder les memes indices pour l'évaluation dans eval.py)
torch.save(test_loader.dataset.indices, "results/test_indices.pth")

#Fonction loss
criterion = nn.CrossEntropyLoss()

#Initialisation des variables pour stocker le meilleur modèle
best_val_loss = float('inf') #initialise à l'infini positif (donc un nbr positif tout simplement), parce que la loss est tjrs >0
best_val_acc = 0
os.makedirs("results/saved_models", exist_ok=True)


#Grid search des hyperparamètres

#Syntaxe pour futur grid search
grid_params = {
    "num_epochs": [50,100],
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

    
      # Optionally, store results for the current configuration
    results_summary.append({
        "config": config_str,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name
})

    #Sauvegarde du meilleur modèle sur base loss et acc (ancien modèle écrasé si meilleur)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "src/resnet18_RGB/results/saved_models/best_loss.pth")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "src/resnet18_RGB/results/saved_models/best_acc.pth")


#Résumé des résultats dans un ficher CSV
df = pd.DataFrame(results_summary)
df.to_csv("src/resnet18_RGB/results/grid_search_results.csv", index=False)
print("CSV résultats.")

# Affichage meilleur modèle
print("\n─── Résumé grid search ───")
for r in sorted(results_summary, key=lambda x: x["val_acc"], reverse=True)[:5]:
    print(f"  {r['config']}  →  val_acc={r['val_acc']:.2f}%  val_loss={r['val_loss']:.4f}")



"""
plt.figure(figsize=(12,5))

plt.suptitle(config_str, fontsize=8)
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Resnet18 - Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Resnet18 - Accuracy')
plt.legend()

plt.show()

"""