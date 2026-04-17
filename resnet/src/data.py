"""On fait des fonctions pour éviter d'avoir des variables gloables qui peuvent poser problèmes"""


import skimage as ski
import matplotlib.pyplot as plt
from pathlib import Path

#Définition des chemins
path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
path_train_val = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/val/"
path_test_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/RGB/"


# Import des données : 600 images train

def load_data_train(path_train_rgb, path_train_val):

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

#x_train, y_train = load_data_train(path_train_rgb, path_train_val)
#print("Nombre images train:", len(y_train))
    

# Import des images tests rgb + labels

def load_data_test(path_test_rgb):
    x_test = []
    for rgb in sorted(Path(path_test_rgb).glob("*.png")): #prends toutes les images qui se terminent par rgb. glob ne marche que sur les objet Path
        x_test.append((ski.io.imread(rgb)))  # Ajoute l'image'
    
    return x_test

x_test = load_data_test(path_test_rgb)
print("Nombre images test:", len(x_test)) #300

#Visualiser les images

labels = ["Health", "Other", "Rust"]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()): #axes.flatten() → transforme la grille en liste : [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
  img = x_train[i]
  ax.imshow(img)
  ax.set_title(labels[y_train[i]])


plt.tight_layout() #pour ajuster les tailles des images et éviter qu'elles se chevauchent
plt.show()