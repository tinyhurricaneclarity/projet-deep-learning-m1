##################################################################################################
                            ### SAISIE des PARAMETRES ###
##################################################################################################

Im_type = ["RGB","MS","HS","RGB_MS","RGB_HS","MS_HS","concat"]                # "RGB","MS","HS","RGB_MS","RGB_HS","MS_HS","concat"

path = "/net/cremi/mvoiturin/projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"
#!ls /content/Kaggle_Prepared/train/RGB


ext = "_augm_F1_CM" #_augm_F1
seuil = {"RGB":0.5,"MS":0.5,"HS":0,"concat":0.6,"RGB_MS":0.57,"RGB_HS":0.57,"MS_HS":0.5}

call_figures = "none"
"""{
    "RGB":["config3_epochs100_cardinality16_learningrate0.001_bwidth3_batch_size32_im_type_RGB"],
    "MS":["config2_epochs100_cardinality16_learningrate0.0001_bwidth5_batch_size32_im_type_MS"],
    "HS":["config0_epochs100_cardinality16_learningrate0.0001_bwidth3_batch_size32_im_type_HS"],
    }"""

path_saved_data = "/net/cremi/mvoiturin/Bureau/projet-deep-learning-m1/ResNext50/saved_model"



#################################################################################################
                                         #fin paramètres#
#################################################################################################



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
import seaborn as sns

#machine learing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
    test_analyse[f"test_F1_{i}"]=[]
    test_analyse[f"moy_F1_{i}"]=0
#analyse : 
    print("")
    print(f"#######  {i}  #######")
    print("")
    
    file = f"{path_saved_data}/resultats_{i}{ext}.json"

    tf = open(file, "r")
    resultats = json.load(tf)
    

    
    for J in resultats :
      
      class_names = resultats[J]["class_names"]
      print(f"{J} :")
      #print(len(resultats[i]["train_accuracies"]), resultats[i]["train_accuracies"])
      #print(len(resultats[i]["train_losses"]), resultats[i]["train_losses"])
      #print(len(resultats[i]["val_accuracies"]), resultats[i]["val_accuracies"])
      #print(len(resultats[i]["val_losses"]), resultats[i]["val_losses"])
      print(f"test_acc : {resultats[J]['test_acc']}")
      print(f"F1-score macro : {resultats[J]['f1_macro']}")
      test_analyse[f"test_acc_{i}"].append(resultats[J]['test_acc'])
      test_analyse[f"test_F1_{i}"].append(resultats[J]['f1_macro'])
      
      epochs = range(1, resultats[J]["epochs"] + 1)
      
      nom_tpr = J.split("_")[0]
      
      for classe in class_names :
          print(f"F1 {classe} : {resultats[J][f'f1_{classe}']}")
      
      
      cm = []
      
      for ind in range(len(class_names)) :
          tpr = []
          for indi in range(len(class_names)):
              tpr.append(resultats[J]["cm"][f"{ind}_{indi}"])
          cm.append(tpr)
          
      print("Matrice de confusion :")
      print(cm) 
      
      if resultats[J]['f1_macro']>seuil[i] :    
          # Accuracy
          plt.figure(figsize=(10,4))
          plt.plot(epochs , resultats[J]["train_accuracies"], label='Train Accuracy')
          plt.plot(epochs, resultats[J]["val_accuracies"], label='Validation Accuracy')
          #plt.plot(epochs, test_accuracies, label='Test Accuracy')
          plt.xlabel("Epoque")
          plt.ylabel("Accuracy")
          plt.title(f"Accuracy par époque pour {nom_tpr}")
          plt.legend()
          plt.show()
        
          # Loss
          plt.figure(figsize=(10,4))
          plt.plot(epochs, resultats[J]["train_losses"], label='Train Loss')
          plt.plot(epochs, resultats[J]["val_losses"], label='Validation Loss')
          #plt.plot(epochs, test_losses, label='Test Loss')
          plt.xlabel("Epoque")
          plt.ylabel("Loss")
          plt.title(f"Loss par époque pour {nom_tpr}")
          plt.legend()
          plt.show()
      
          plt.figure(figsize=(8, 8))
          sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
          plt.title('Confusion Matrix')
          plt.ylabel('True label')
          plt.xlabel('Predicted label')
          plt.show()
      
      print("")
    #meilleurs paramètres :
    
    
    
    nom = "deb"
    best_accuracy = 0
    
    
    for J in resultats :
      if nom == "deb" or resultats[J]["test_acc"]>best_accuracy:
        nom = J
        best_accuracy = resultats[J]["test_acc"]
    
       
    
    test_analyse[f"best_test_acc_{i}"]=(nom,best_accuracy)
    
    if call_figures!="none" :
        for h in call_figures[i] : 
           plt.figure(figsize=(10,4))
           plt.plot(epochs , resultats[h]["train_accuracies"], label='Train Accuracy')
           plt.plot(epochs, resultats[h]["val_accuracies"], label='Validation Accuracy')
           #plt.plot(epochs, test_accuracies, label='Test Accuracy')
           plt.xlabel("Epoque")
           plt.ylabel("Accuracy")
           plt.title(f"Accuracy par époque pour {nom_tpr}")
           plt.legend()
           plt.show()
         
           # Loss
           plt.figure(figsize=(10,4))
           plt.plot(epochs, resultats[J]["train_losses"], label='Train Loss')
           plt.plot(epochs, resultats[J]["val_losses"], label='Validation Loss')
           #plt.plot(epochs, test_losses, label='Test Loss')
           plt.xlabel("Epoque")
           plt.ylabel("Loss")
           plt.title(f"Loss par époque pour {nom_tpr}")
           plt.legend()
           plt.show()
      
    
    
    
    
    best_val_accuracy = 0
    nom = "deb"
    for J in resultats :
      if nom == "deb" or resultats[J]["best_val_acc"]>best_val_accuracy:
        nom = J
        best_val_accuracy = resultats[J]["best_val_acc"]
    
    test_analyse[f"Best_at_t_test_acc_{i}"] = [nom,best_val_accuracy,resultats[nom]['best_epoch']]
    test_analyse[f"moy_test_acc_{i}"]=sum(test_analyse[f'test_acc_{i}'])/len(test_analyse[f'test_acc_{i}'])
    test_analyse[f"moy_F1_{i}"]=sum(test_analyse[f'test_F1_{i}'])/len(test_analyse[f'test_F1_{i}'])
    
print("")
for i in Im_type :
    print(f"la moyenne des acc_test pour {i} est de {test_analyse[f'moy_test_acc_{i}']}")
    print(f"la moyenne des F1-score pour {i} est de {test_analyse[f'moy_F1_{i}']}")   
    print(f"meilleur test accuracy générale pour {test_analyse[f'best_test_acc_{i}'][0]} : {test_analyse[f'best_test_acc_{i}'][1]}")
    print(f"meilleur valeur accuracy à un temps donné pour {test_analyse[f'Best_at_t_test_acc_{i}'][0]} : {test_analyse[f'Best_at_t_test_acc_{i}'][1]} à l'époque {test_analyse[f'Best_at_t_test_acc_{i}'][2]}")
    print("")