"""se placer dans resnet18
Permet de créer un csv contenant les .shape de chaque image
"""

import skimage as ski
import pandas as pd
import os

path_train_hs = "/autofs/unityaccount/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/HS"

def load_data_train(path_train_hs):

    x_train = []
    y_train = []

    for i in range(1, 201):

        path_health = f"{path_train_hs}/Health_hyper_{i}.tif"
        path_other  = f"{path_train_hs}/Other_hyper_{i}.tif"
        path_rust   = f"{path_train_hs}/Rust_hyper_{i}.tif"

        x_train.append(ski.io.imread(path_health))
        x_train.append(ski.io.imread(path_other))
        x_train.append(ski.io.imread(path_rust))

        y_train.append(0)
        y_train.append(1)
        y_train.append(2)

    return x_train, y_train


x_train, y_train = load_data_train(path_train_hs)


def shape(x_train):

    data = []

    for i, img in enumerate(x_train):

        data.append({
            "index":i, 
            "shape":img.shape
        })
    
    df = pd.DataFrame(data)
    return df


df = shape(x_train)
print(df.head())
os.makedirs("src/resnet18_HS/results", exist_ok=True)
df.to_csv("src/resnet18_HS/results/visualisation_bandes_HS.csv", index=False)


"""POur visualiser la dernière bande (126)"""
path = "/autofs/unityaccount/cremi/leanguye/projet-deep-learning-m1/resnet18/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/HS/Rust_hyper_200.tif"

img = ski.io.imread(path)

print(img.shape)

last_band = img[:, :, -1]

print(last_band.min())
print(last_band.max())
print(last_band.mean())

"""(32, 32, 126)
65535
65535
65535.0
le min max et mean sont pareils, ce qui signifie que la bande 126 ne comporte pas de données intéressantes
"""