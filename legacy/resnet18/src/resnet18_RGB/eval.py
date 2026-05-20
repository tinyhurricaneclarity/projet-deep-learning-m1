#Evaluation test set

"""RESNET18 Se placer dans le dossier resnet18 pour lancer le code"""
"""Fichier éval permet de recreéer le split du dataset utilisé pour l'entraintement du modèle. 
Les indices de split du dataset sont sauvegardés au cours de l'évaluation et le dataset est resplitté selon les indices dans ce fichier.
Le modèle testé est le meilleur selon la val loss/acc"""

import model
import torch
import data_load
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from config import class_names, batch_size
import numpy as np
#ne pas faire from train import car quand Python importe depuis train.py, il exécute tout le fichier train.py — y compris la boucle d'entraînement.
#sinon ajouter dans train.py if __name__ == "__main__":

#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet50().to(device) #lance une nouvelle instance du modèle, ca réinitialise tout.
print(model)

#Chargement du modèle sauvegardé à tester
model.load_state_dict(torch.load("src/resnet50_RGB/results/saved_models/best_acc_RGB_data_aug_50.pth"))

#Import du test dataset (meme prcoédure que dans train, appliqué sur test)

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"


class_names = ["Health", "Rust", "Other"]
Im_type     = "RGB"
Num_data    = 600  # 200 images par classe

split = torch.load("src/resnet50_RGB/results/split_par_classe_RGB_data_aug_50.pth") #dictionnaire qui renvoit {"test":(classe, indice), val..., train...}
test  = data_load.import_images(class_names, split, "test") #renvoie uniquement les tuples de "test"
test_dataset = data_load.CustomImageDataset(test['images'], test['labels'], transform=data_load.eval_transform)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


print("Evaluation sur le jeu de test\n")

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



# Accuracy, recall, precision

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda y, p: precision_score(y, p, average="macro"),
    "Recall": lambda y, p: recall_score(y, p, average="macro"), 
    "F1-score macro": lambda y, p: f1_score(y, p, average="macro") 
}

for name, function in metrics.items():
    score = function(all_labels, all_preds)
    print(f"{name} : {score:.4f}%")


# F1-score par classe
f1_par_classe = f1_score(all_labels, all_preds, average=None)
for i, classe in enumerate(class_names):
    print(f"F1-score {classe} : {f1_par_classe[i]:.4f}")

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print("Matrice de confusion :")
print(cm)


