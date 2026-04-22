#Evaluation test set

import model
import torch
import data_load
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from config import class_names, batch_size
#ne pas faire from train import car quand Python importe depuis train.py, il exécute tout le fichier train.py — y compris la boucle d'entraînement.
#sinon ajouter dans train.py if __name__ == "__main__":

#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet18().to(device) #lance une nouvelle instance du modèle, ca réinitialise tout.
print(model)

#Chargement du modèle sauvegardé à tester
model.load_state_dict(torch.load("saved_models/convnet_best.pth"))

#Import des datasets train, val et test 

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
x_train, y_train = data_load.load_data_train(path_train_rgb)

#Convertion en tensor et trainloaderpourq
dataset = data_load.CustomImageDataset(x_train, y_train, transform=None)

test_indices = torch.load("results/test_indices.pth")
test_dataset = torch.utils.data.Subset(dataset, test_indices) #crée un sous-ensemble du dataset original en utilisant une liste d’indices.
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #créer les batch sans shuffle à partir du test dataset

print("Evaluation sur le jeu de test.")
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

# F1-score global
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"F1-score macro : {f1:.4f}")

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print("Matrice de confusion :")
print(cm)

# F1-score par classe
f1_par_classe = f1_score(all_labels, all_preds, average=None)
for i, classe in enumerate(class_names):
    print(f"F1-score {classe} : {f1_par_classe[i]:.4f}")

