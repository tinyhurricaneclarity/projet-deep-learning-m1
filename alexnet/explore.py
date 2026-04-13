# Lecture et comptage des fichiers dans RGB 

import os

data_path = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB"

# Lister tous les fichiers
fichiers = os.listdir(data_path)

# Compter par classe
health = [f for f in fichiers if f.startswith("Health")]
rust   = [f for f in fichiers if f.startswith("Rust")]
other  = [f for f in fichiers if f.startswith("Other")]

print("Total images :", len(fichiers))
print("Health :", len(health))
print("Rust :", len(rust))
print("Other :", len(other))




# On obtient : 
# Total images : 600
# Health : 200
# Rust : 200
# Other : 200
