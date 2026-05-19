##################################################################################################
                            ### SAISIE des PARAMETRES ###
##################################################################################################

#liste des noms de classes :
class_names = ["Health","Rust","Other"] # classes présentes

#type d'image :
Im_type = "concat"                 # "RGB","MS,"HS"

#Nombre d'image :
Num_data = 600           # Nombre d'images dans les datas. Normalement, c'est un multiple du nomùbre de classe ayant un jeu de datas parfaitement équilibré

#proportion de chaque partie
proportion_train = 70        # proportion des datas à mettre dans train en % ou proportion
proportion_validation = 10
proportion_test = 20              # proportion des datas à mettre dans test en % ou proportion

#ils sont juste là.
n_test = 99
n_val = 99

#chemin vers le dossier ontenant les images devant contenir le dossier RGB, HS ou MS (ou les trois) contenant les images à analyser
path = "/net/cremi/mvoiturin/projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"
#!ls /content/Kaggle_Prepared/train/RGB

#Hyperparamètres de ResNext50
#On retient ces hyperparamètres, mais il y en a d'autres
grid_params = {
   "num_epochs": [100],#
   "cardinality" : [16,32,48], # 16,32,48
   "learning_rate": [0.001,0.0001],#0.001,0.0001
   "bwidth" : [3,4,5],#3,4,5
   "batch_size": [32],#"32,"
   #"num_blocks": [ 10,12],#"10,"
   #"base_channels": [ 32,64],#"32,"
   "kernel_size": [3]#", 5"

}

#seuil d'enregistrement du modèle. A 1 ou supérieur : n'enregistre pas de modèles.
seuil = 1

#extention de nom d'enregistrement du dictionnaire résumant le grid search
spec_para = "_augm_F1_CM"
#chemin vers le dossier de sauvegardes:
path_saved_data = "/net/cremi/mvoiturin/Bureau/projet-deep-learning-m1/ResNext50/saved_model"


# etape 4 Si etape ne contien pas 4, ne sert à rien.
#chemin vers le dossier du modèle sauvegardé :
fichier = "/net/cremi/mvoiturin/Bureau/projet-deep-learning-m1/ResNext50/saved_model"
#nom du modèle sauvegardé et indices :
nom_modele = "modelconfig0_epochs100_cardinality16_learningrate0.0001_bwidth3_batch_size32_im_type_RGB_MS_HS_epoch100.pth"
ind_cardinality_modele = 16
ind_bwidth_modele = 3


#etapes
etapes = [1,2,3] # de 1 à 3
#seed de Random
seed_id = 42



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



###classe RESNEXT ###
#ResNext block : 
class resnext_block(nn.Module):

  def __init__(self, in_channels, cardinality, bwidth, idt_downsample=None, stride=1):
    super(resnext_block, self).__init__()
    self.expansion = 2
    out_channels = cardinality * bwidth
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=cardinality, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = idt_downsample

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
        identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

#ResNext :
class ResNeXt(nn.Module):

  def __init__(self, resnet_block, layers, cardinality, bwidth, img_channels, num_classes):
    super(ResNeXt, self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.cardinality = cardinality
    self.bwidth = bwidth
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # ResNeXt Layers
    self.layer1 = self._layers(resnext_block, layers[0], stride=1)
    self.layer2 = self._layers(resnext_block, layers[1], stride=2)
    self.layer3 = self._layers(resnext_block, layers[2], stride=2)
    self.layer4 = self._layers(resnext_block, layers[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(self.cardinality * self.bwidth, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x

  def _layers(self, resnext_block, no_residual_blocks, stride):
    identity_downsample = None
    out_channels = self.cardinality * self.bwidth
    layers = []

    if stride != 1 or self.in_channels != out_channels * 2:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*2, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*2))

      layers.append(resnext_block(self.in_channels,  self.cardinality, self.bwidth, identity_downsample, stride))
    self.in_channels = out_channels * 2

    for i in range(no_residual_blocks - 1):
      layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth))

    self.bwidth *= 2

    return nn.Sequential(*layers)

  def _layers(self, resnext_block, no_residual_blocks, stride):
        identity_downsample = None
        out_channels = self.cardinality * self.bwidth
        layers = []

        if stride != 1 or self.in_channels != out_channels * 2:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*2, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*2))

        layers.append(resnext_block(self.in_channels,  self.cardinality, self.bwidth, identity_downsample, stride))
        self.in_channels = out_channels * 2

        for i in range(no_residual_blocks - 1):
            layers.append(resnext_block(self.in_channels, self.cardinality, self.bwidth))

        self.bwidth *= 2

        return nn.Sequential(*layers)

#ResNext50 :
def ResNeXt50(img_channels=3, num_classes=1000, cardinality=32, bwidth=4):
    return ResNeXt(resnext_block, [3,4,6,3],  cardinality, bwidth, img_channels, num_classes)



###FONCTIONS###

# Import des images tests rgb + labels

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

config_counter = 0
dico_config = {}

for (num_epochs, cardinality, learning_rate, batch_size, bwidth) in itertools.product(
     grid_params["num_epochs"],
     grid_params["cardinality"],
     grid_params["learning_rate"],
     grid_params["batch_size"],
     #grid_params["num_blocks"],
     #grid_params["base_channels"],
     grid_params["bwidth"]):
 
   #on fait un dicionnaire config pour que ce soit plus simple
   config = {
       "num_epochs": num_epochs,
       "learning_rate": learning_rate,
       "batch_size": batch_size,
       #"num_blocks": num_blocks,
       #"base_channels": base_channels,
       "bwidth": bwidth,
       "cardinality": cardinality
    }
 
 #POur suivre les grid search
   config_str = f"config{config_counter}_epochs{num_epochs}_cardinality{cardinality}_learningrate{learning_rate}_bwidth{bwidth}_batch_size{batch_size}_im_type_{Im_type}"
   dico_config[config_str] = config
   config_counter += 1
   print(config_str)
print("")

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


def import_images(class_names, dico, Im_type, tr_or_te="train"):
    """
    Charge et fusionne les images RGB, MS et HS pour chaque image.
    RGB : 3 canaux (png) 64x64
    MS  : 5 canaux (tif) 64x64
    HS  : 125 canaux (tif) 32x32 -> redimensionné en 64x64
    Concaténation -> 133 canaux au total
    """
    
    images = []
    labels = []
    indices = dico[tr_or_te]
    
    if Im_type in ["MS","RGB"]:
        for (j, i) in indices:
            tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
            img = ski.io.imread(tpr_path)
            img = img.astype(np.float32)  # conversion en float32
            # normalisation entre 0 et 1
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            images.append(img)
            labels.append(j)
        return {"images": np.array(images), "labels": labels}
    
    elif Im_type == "HS" :
        for (j, i) in indices:
            tpr_path = f"{dico['path_data']}/{class_names[j]}_hyper_{i}{dico['sufix']}"
            img = ski.io.imread(tpr_path)
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            
            # Force 125 canaux : certaines images HS peuvent avoir un nombre différent
            if img.shape[2] < 125:
                # padding avec des zéros si moins de 125 canaux
                pad = np.zeros((32, 32, 125 - img.shape[2]), dtype=np.float32)
                img = np.concatenate([img, pad], axis=2)
            elif img.shape[2] > 125:
                # coupe si plus de 125 canaux
                img = img[:, :, :125]
            
            images.append(img)
            labels.append(j)
        return {"images": np.array(images), "labels": labels}
    
    else :
        for (j, i) in indices:
            # Chargement RGB (3 canaux, 64x64)
            path_rgb = f"{dico['path_data_rgb']}/{class_names[j]}_hyper_{i}.png"
            img_rgb  = ski.io.imread(path_rgb).astype(np.float32) / 255.0
    
            # Chargement MS (5 canaux, 64x64)
            path_ms = f"{dico['path_data_ms']}/{class_names[j]}_hyper_{i}.tif"
            img_ms  = ski.io.imread(path_ms).astype(np.float32)
            img_ms  = (img_ms - img_ms.min()) / (img_ms.max() - img_ms.min() + 1e-6)
    
            # Chargement HS (125 canaux, 32x32 -> redimensionné en 64x64)
            path_hs = f"{dico['path_data_hs']}/{class_names[j]}_hyper_{i}.tif"
            img_hs  = ski.io.imread(path_hs).astype(np.float32)
            img_hs  = (img_hs - img_hs.min()) / (img_hs.max() - img_hs.min() + 1e-6)
    
            # Force 125 canaux
            if img_hs.shape[2] < 125:
                pad = np.zeros((img_hs.shape[0], img_hs.shape[1], 125 - img_hs.shape[2]), dtype=np.float32)
                img_hs = np.concatenate([img_hs, pad], axis=2)
            elif img_hs.shape[2] > 125:
                img_hs = img_hs[:, :, :125]
    
            # Redimensionnement HS 32x32 -> 64x64
            img_hs = resize(img_hs, (64, 64, 125), anti_aliasing=True).astype(np.float32)
                    
            if Im_type=="RGB_MS":
                    # Concaténation RGB + MS
                img_fusion = np.concatenate([img_rgb, img_ms], axis=2)
                images.append(img_fusion)
                labels.append(j)
            elif Im_type=="RGB_HS":
                # Concaténation RGB + HS
                img_fusion = np.concatenate([img_rgb, img_hs], axis=2)
                images.append(img_fusion)
                labels.append(j)
            elif Im_type=="MS_HS":
                # Concaténation RGB + MS
                img_fusion = np.concatenate([img_ms,img_hs], axis=2)
                images.append(img_fusion)
                labels.append(j)
            else :
                # Concaténation RGB + MS + HS -> 133 canaux
                img_fusion = np.concatenate([img_rgb, img_ms, img_hs], axis=2)
                images.append(img_fusion)
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5], # 5 canaux en MS !!! valeurs génériques. 
                         std=[0.5, 0.5, 0.5, 0.5, 0.5]) #permet de "lisser" la distribution des tensors (valeurs autour de 0) pour accélérer l’entraînement et rendre l’optimisation plus stable
])


# Pour val et test : sans augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5, 0.5, 0.5])
])


# Classe CustomImageDataset :
"""
Création du dataset avec image/label en tensor (compatibilité resnet)
"""
#demande en argument une liste d'image et les labels (sous forme numériques) qui y sont liés.
#exemple : CustomImageDataset(liste_images,liste_labels)

to_tensor = transforms.ToTensor()





### PROGRAME ###

if 1 in etapes:
    
        
    random.seed(seed_id)
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    
    # fabrication de Train et test de manière aléatoire pour construire le modèle :
    dico_train_test = alea_train_test(Num_data, class_names, n_val=n_val, n_test=n_test)
    if Im_type in ["HS","MS","RGB"]:
        sufix_and_path(Im_type, dico_train_test, path)

    # Chemins vers les trois modalités
    dico_train_test["path_data_rgb"] = f"{path}/RGB"
    dico_train_test["path_data_ms"]  = f"{path}/MS"
    dico_train_test["path_data_hs"]  = f"{path}/HS"
    

    # Chargement des images
    Train = import_images(class_names, dico_train_test, Im_type, "train",)
    Val   = import_images(class_names, dico_train_test, Im_type, "val")
    Test  = import_images(class_names, dico_train_test, Im_type, "test")
    
    print(f"Train : {len(Train['images'])} images")
    print(f"Val   : {len(Val['images'])} images")
    print(f"Test  : {len(Test['images'])} images")
    
    # sauvegarde test loader indices (pour garder les memes indices pour l'évaluation dans eval.py)
    #torch.save(dico_train_test, "src/resnet18_RGB/results/split_par_classe.pth") #dictionnaire {"images": np.array(images), "labels": labels}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    
    criterion = nn.CrossEntropyLoss()
    #Convertion en tensor et trainloader
    
    
    ### DATASETS ET DATALOADERS
    
    batch_size = 32
    
    train_dataset = CustomImageDataset(Train['images'], Train['labels'], transform=None)
    val_dataset   = CustomImageDataset(Val['images'],   Val['labels'],   transform=None)
    test_dataset  = CustomImageDataset(Test['images'],  Test['labels'],  transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    
    
    #______________________________________________________________________________________________________________________________________________
    #Boucle entrainement du modèle
    #stockage de résultats
    resultats = {}
    
    
    if 2 in etapes :
        for i in dico_config :
          print(f"Running config : {i}")
          config = dico_config[i]
          resultats[i]= {"train_accuracies" : [], "train_losses" : [], "val_accuracies" : [], "val_losses" : [], "epochs" : config["num_epochs"]}
        
          #Création instance du Resnext avec les valeurs d'hyperparamètres du grid search
        
          if Im_type == "RGB" :
            tpr = 3
          elif Im_type == "MS" :
            tpr = 5
          elif Im_type == "HS" :
            tpr = 125
          elif Im_type == "RGB_MS":
            tpr = 8
          elif Im_type == "RGB_HS":
            tpr = 128
          elif Im_type == "MS_HS":
            tpr = 130
          else :
            tpr  = 133
        
          model = ResNeXt50(img_channels=tpr, num_classes= len(class_names), cardinality=config["cardinality"], bwidth=config["bwidth"])
          optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
          #Boucle entrainement du modèle
        
          for epoch in range(1, config["num_epochs"] + 1):
            model.train() #on le met en mode train (MAJ des gradident)
            running_loss = 0.0 # On initialise une variable pour accumuler la perte sur tous les batches de l’époque.
            correct_train = 0
            total_train = 0
        
        
            #Ces variables servent à compter le nombre de prédictions correctes et le nombre total d’exemples pour calculer l’accuracy sur l’ensemble d’entraînement.
            #accuracy = correct_train / total_train
        
        
            #on donne les images par batch au modèle
            for images, labels in train_loader:
              images, labels = images.to(device), labels.to(device) #Déplace les images et les labels sur le même périphérique que le modèle (CPU ou GPU).
              #Important : tous les tenseurs doivent être sur le même device pour que les opérations fonctionnent.
        
              labels = labels.squeeze() ##ensure labels are of shape[batch_size]
              #IMPORTANT, en pytorch, les labels ne sont pas des str mais des tensors contenant des nombres entiers
        
        
              optimizer.zero_grad() #on remet les gradients à 0. Sinon Pytorch accumule avec les anciens gradients
        
        
              outputs = model(images) #on train le modele sur les images (forward)
              #Résultat : un tensor de shape (batch_size, num_classes), avec les scores de chaque classe pour chaque image.
        
              #calcul des fonctions loss
              loss = criterion(outputs, labels) #calcul la valeur de la perte moyenne du batch
        
              #backpropagation
              loss.backward() #calcul les gradients de tous les paramètres (coef directeur)
              #Calculates the slope at the current position. It determines how each parameter is contributing to the error and in which direction the model should move to reduce it
        
              optimizer.step() #met à jour les poids à partir des gradients calculés. Uses that calculated slope to update the parameters. It performs the actual movement toward a lower loss value.
        
              running_loss += loss.item() * images.size(0)
              #loss est un tensor PyTorch qui contient la valeur de la perte moyenne pour le batch, définit plus haut avec loss = criterion...
              #.item() permet de transformer un tensor en nombre python (float)
              # perte totale = perte moyenne par exemple×taille du batch
        
        
              #calcul de l'accuracy (nombre de bonnes prédictions)
              _, predicted = torch.max(outputs, 1) #explications en bas. Permet de prendre la classe avec le score le plus élevé
              total_train += labels.size(0) # labels.size(0) = nombre d’images dans le batch (batch_size). Total_train permet de suivre le nombre d'images traitées au total après le passage de chaque batch.
              #size(0) parce qu'on demande la taille de la dimension 0 du tensorn donc le nombre d'image
        
              correct_train += (predicted == labels).sum().item()
        
            train_loss = running_loss / len(train_loader.dataset)
            resultats[i]["train_losses"].append(train_loss)
            train_acc = correct_train / total_train
            resultats[i]["train_accuracies"].append(train_acc)
        
            #Validation loop
            model.eval() #mode évaluation
            correct_val = 0
            total_val = 0
            running_val_loss = 0.0
        
            with torch.no_grad():
              for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze()
                outputs = model(images)
        
                loss_val = criterion(outputs, labels)
                running_val_loss += loss_val.item() * images.size(0)
        
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
            val_acc = correct_val / total_val
            resultats[i]["val_accuracies"].append(val_acc)
            val_loss = running_val_loss / len(val_loader.dataset)
            resultats[i]["val_losses"].append(val_loss)
        
            print(f"{i} | Epoch [{epoch}/{config['num_epochs']}] | Train Loss {train_loss:.4f} [ Train Acc:{train_acc:.4f} | Val Acc: {val_acc:.4f}] | Val Loss {val_loss:.4f}")
        
        
           # Final test evaluation for this configuration.
          model.eval()
          correct_test = 0
          total_test = 0
          with torch.no_grad():
              for images, labels in test_loader:
                  images, labels = images.to(device), labels.to(device)
                  labels = labels.squeeze()   # Ensure labels are of shape [batch_size]
                  outputs = model(images)
                  _, predicted = torch.max(outputs, 1)
                  total_test += labels.size(0)
                  correct_test += (predicted == labels).sum().item()
        
          test_acc = correct_test / total_test
          resultats[i]["test_acc"] = test_acc
          print(f"{i} | Test Accuracy: {test_acc:.4f}")
        
          best_val_acc = max(resultats[i]["val_accuracies"])
          best_epoch = resultats[i]["val_accuracies"].index(best_val_acc)
        
          resultats[i]["best_val_acc"] = best_val_acc
          resultats[i]["best_epoch"] = best_epoch
        
          
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

          F1_sco = f1_score(all_labels, all_preds, average=None)
          
          
          resultats[i]["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
          for ind in range(len(class_names)) :
              resultats[i][f"f1_{class_names[ind]}"] =  F1_sco[ind]
          
          cm = confusion_matrix(all_labels, all_preds)
          resultats[i][f"cm"] = {}
          
          for ind in range(len(cm)) :  
              for indi in range(len(cm[ind])):
                  resultats[i]["cm"][f"{ind}_{indi}"] = int(cm[ind][indi])
          
          resultats[i]["class_names"]=class_names
            
          if test_acc>seuil :
            #Stocker le modèle à la fin de chaque epoch pour pouvoir comparer les loss entre les hyperparamètres
            
            if Im_type=="RGB" :
              os.makedirs(f"{path_saved_data}/RGB", exist_ok=True)
              file = f"{path_saved_data}/RGB"
            elif Im_type=="MS" :
              os.makedirs(f"{path_saved_data}/MS", exist_ok=True)
              file = f"{path_saved_data}/MS"
            elif Im_type=="HS" :
              os.makedirs(f"{path_saved_data}/HS", exist_ok=True)
              file = f"{path_saved_data}/HS"
            elif Im_type=="RGB_MS" :
              os.makedirs(f"{path_saved_data}/RGB_MS", exist_ok=True)
              file = f"{path_saved_data}/RGB_MS"
            elif Im_type=="RGB_HS" :
              os.makedirs(f"{path_saved_data}/RGB_HS", exist_ok=True)
              file = f"{path_saved_data}/RGB_HS"
            elif Im_type=="MS_HS" :
              os.makedirs(f"{path_saved_data}/MS_HS", exist_ok=True)
              file = f"{path_saved_data}/MS_HS"
            else :
              os.makedirs(f"{path_saved_data}/concat", exist_ok=True)
              file = f"{path_saved_data}/concat"
            
            model_save_path = os.path.join(file,f"model{i}_epoch{config['num_epochs']}.pth")
            
            if 4 in etapes :
                nom_modele = f"model{i}_epoch{config['num_epochs']}.pth"
                ind_cardinality_modele = config["cardinality"]
                ind_bwidth_modele = config["bwidth"]

            
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
          print("")
        """ QUELQUES EXPLICATIONS
        
        correct_train += (predicted == labels).sum().item()
        (predicted == labels) : crée un tensor booléen de la forme (batch_size,).
        True si la prédiction est correcte, False sinon.
        .sum() : compte le nombre de True (nombre de bonnes prédictions).
        .item() : convertit le tensor en nombre Python.
        correct_train cumule toutes les bonnes prédictions jusqu’ici.
        
         predicted = torch.tensor([0, 1])
        labels = torch.tensor([0, 2])
        (predicted == labels)  # tensor([True, False])
        (predicted == labels).sum().item()  # 1"""
        if 3 in etapes :
                
            if Im_type=="RGB" :
              file = f"{path_saved_data}/resultats_RGB{spec_para}.json"
            elif Im_type=="MS" :
              file = f"{path_saved_data}/resultats_MS{spec_para}.json"
            elif Im_type=="HS" :
              file = f"{path_saved_data}/resultats_HS{spec_para}.json"
            elif Im_type=="RGB_MS" :
              file = f"{path_saved_data}/resultats_RGB_MS{spec_para}.json"
            elif Im_type=="RGB_HS" :
              file = f"{path_saved_data}/resultats_RGB_HS{spec_para}.json"
            elif Im_type=="MS_HS" :
              file = f"{path_saved_data}/resultats_MS_HS{spec_para}.json"
            else : 
              file = f"{path_saved_data}/resultats_concat{spec_para}.json"
                
            tf = open(file, "w")
            json.dump(resultats, tf)
            tf.close()
    
    
if 4 in etapes :
    path_modele = f"{fichier}/{Im_type}/{nom_modele}"


    ### EVALUATION FINALE SUR LE JEU DE TEST
    if Im_type == "RGB" :
      tpr = 3
    elif Im_type == "MS" :
      tpr = 5
    elif Im_type == "HS" :
      tpr = 125
    elif Im_type == "RGB_MS":
      tpr = 8
    elif Im_type == "RGB_HS":
      tpr = 128
    elif Im_type == "MS_HS":
      tpr = 130
    else :
      tpr = 133
    
    model = ResNeXt50(img_channels=tpr, num_classes= len(class_names), cardinality=ind_cardinality_modele , bwidth=ind_bwidth_modele)

    
    print("\nChargement du meilleur modèle.")
    model.load_state_dict(torch.load(path_modele))

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

    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"F1-score macro : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Matrice de confusion :")
    print(cm)

    f1_par_classe = f1_score(all_labels, all_preds, average=None)
    for i, classe in enumerate(class_names):
        print(f"F1-score {classe} : {f1_par_classe[i]:.4f}")


    print("\nTerminé.")
