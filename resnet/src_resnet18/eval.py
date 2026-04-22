#Evaluation test set

import model
import torch
import data_load
from train import batch_size
from data_load import create_dataloader
from torch.utils.data import DataLoader

#Définition du device et du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.ResNet18().to(device) #lance une nouvelle instance du modèle, ca réinitialise tout.
print(model)

#Chargement du modèle sauvegardé à tester
model.load_state_dict(torch.load("saved_models/convnet_best.pth"))

#Import des datasets train, val et test 

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
x_train, y_train = data_load.load_data_train(path_train_rgb)

#Convertion en tensor et trainloader
dataset = data_load.CustomImageDataset(x_train, y_train, transform=None)

test_indices = torch.load("results/test_indices.pth")
test_dataset = torch.utils.data.Subset(dataset, test_indices)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print("Evaluation sur le jeu de test.")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images       = images.to(device)
        labels       = labels.squeeze()
        outputs      = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

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

