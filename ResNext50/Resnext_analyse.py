#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:42:57 2026

@author: mvoiturin
"""

### SAISIE des PARAMETRES ###

class_names = ["Health","Rust","Other"] # classes présentes
Im_type = ["RGB","MS","HS","concat"]                # "RGB","MS","HS"

path = "/net/cremi/mvoiturin/projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"
#!ls /content/Kaggle_Prepared/train/RGB


ext = "_augm_F1"
seuil = {"RGB":0.5,"MS":0.5,"HS":0.5,"concat":0.5}

call_figures = "none"
"""{
    "RGB":["config3_epochs100_cardinality16_learningrate0.001_bwidth3_batch_size32_im_type_RGB"],
    "MS":["config2_epochs100_cardinality16_learningrate0.0001_bwidth5_batch_size32_im_type_MS"],
    "HS":["config0_epochs100_cardinality16_learningrate0.0001_bwidth3_batch_size32_im_type_HS"],
    }"""

path_saved_data = "/net/cremi/mvoiturin/Bureau/projet-deep-learning-m1/ResNext50/saved_model"

##IMPORT DE LIBRARY
# tableaux, datas, images et gestion de fichiers :
import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path
from skimage.transform import resize
from collections import Counter

#machine learing
from sklearn.metrics import precision_score, recall_score, f1_score

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optim


####################################################################################################################
                                         # Graph et figures #
####################################################################################################################

test_analyse = {}  


for i in Im_type :
    test_analyse[f"test_acc_{i}"]=[]
    test_analyse[f"moy_test_acc_{i}"]=0
#analyse : 
    
    if i=="RGB" :
      file = f"{path_saved_data}/resultats_RGB{ext}.json"
    elif i=="MS" :
      file = f"{path_saved_data}/resultats_MS{ext}.json"
    elif i=="HS" :
      file = f"{path_saved_data}/resultats_HS{ext}.json"
    else:
      file = f"{path_saved_data}/resultats_concat{ext}.json"
        
    tf = open(file, "r")
    resultats = json.load(tf)
    
    
    for J in resultats :
      print(f"{J} :")
      #print(len(resultats[i]["train_accuracies"]), resultats[i]["train_accuracies"])
      #print(len(resultats[i]["train_losses"]), resultats[i]["train_losses"])
      #print(len(resultats[i]["val_accuracies"]), resultats[i]["val_accuracies"])
      #print(len(resultats[i]["val_losses"]), resultats[i]["val_losses"])
      print(f"test_acc : {resultats[J]['test_acc']}")
      test_analyse[f"test_acc_{i}"].append(resultats[J]['test_acc'])
      
      epochs = range(1, resultats[J]["epochs"] + 1)
      
      if resultats[J]['test_acc']>seuil[i] :    
          # Accuracy
          plt.figure(figsize=(10,4))
          plt.plot(epochs , resultats[J]["train_accuracies"], label='Train Accuracy')
          plt.plot(epochs, resultats[J]["val_accuracies"], label='Validation Accuracy')
          #plt.plot(epochs, test_accuracies, label='Test Accuracy')
          plt.xlabel("Epoch")
          plt.ylabel("Accuracy")
          plt.title("Accuracy par epoch")
          plt.legend()
          plt.show()
        
          # Loss
          plt.figure(figsize=(10,4))
          plt.plot(epochs, resultats[J]["train_losses"], label='Train Loss')
          plt.plot(epochs, resultats[J]["val_losses"], label='Validation Loss')
          #plt.plot(epochs, test_losses, label='Test Loss')
          plt.xlabel("Epoch")
          plt.ylabel("Loss")
          plt.title("Loss par epoch")
          plt.legend()
          plt.show()
    
    if call_figures!="none" :
        for j in call_figures[i] : 
           plt.figure(figsize=(10,4))
           plt.plot(epochs , resultats[J]["train_accuracies"], label='Train Accuracy')
           plt.plot(epochs, resultats[J]["val_accuracies"], label='Validation Accuracy')
           #plt.plot(epochs, test_accuracies, label='Test Accuracy')
           plt.xlabel("Epoch")
           plt.ylabel("Accuracy")
           plt.title("Accuracy par epoch")
           plt.legend()
           plt.show()
         
           # Loss
           plt.figure(figsize=(10,4))
           plt.plot(epochs, resultats[J]["train_losses"], label='Train Loss')
           plt.plot(epochs, resultats[J]["val_losses"], label='Validation Loss')
           #plt.plot(epochs, test_losses, label='Test Loss')
           plt.xlabel("Epoch")
           plt.ylabel("Loss")
           plt.title("Loss par epoch")
           plt.legend()
           plt.show()
      
    print("")
    
    print("Evaluation sur le jeu de test.")

    print(f"F1-score macro : {resultats[i]['f1']:.4f}")
    
    print("Matrice de confusion :")
    print(resultats[i]["cm"])

    print("")
    
    for classe in class_names :
        print(resultats[i][f"F1-score_{classe}"])
    
    #meilleurs paramètres :
    
    nom = "deb"
    best_accuracy = 0
    
    for J in resultats :
      if nom == "deb" or resultats[J]["test_acc"]>best_accuracy:
        nom = J
        best_accuracy = resultats[J]["test_acc"]
    
    test_analyse[f"best_test_acc_{i}"]=(nom,best_accuracy)
    
    
    
    best_val_accuracy = 0
    nom = "deb"
    for J in resultats :
      if nom == "deb" or resultats[J]["best_val_acc"]>best_val_accuracy:
        nom = J
        best_val_accuracy = resultats[J]["best_val_acc"]
    
    test_analyse[f"Best_at_t_test_acc_{i}"] = [nom,best_val_accuracy,resultats[nom]['best_epoch']]
    test_analyse[f"moy_test_acc_{i}"]=sum(test_analyse[f'test_acc_{i}'])/len(test_analyse[f'test_acc_{i}'])

print("")
for i in Im_type :
    print(f"la moyenne des acc_test pour {i} est de {test_analyse[f'moy_test_acc_{i}']}")
    print(f"meilleur test accuracy générale pour {test_analyse[f'best_test_acc_{i}'][0]} : {test_analyse[f'best_test_acc_{i}'][1]}")
    print(f"meilleur valeur accuracy à un temps donné pour {test_analyse[f'Best_at_t_test_acc_{i}'][0]} : {test_analyse[f'Best_at_t_test_acc_{i}'][1]} à l'époque {test_analyse[f'Best_at_t_test_acc_{i}'][2]}")
    print("")