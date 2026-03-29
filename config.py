# config.py

# Chemin vers le dossier principal des données
ROOT = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared"

# Sous-dossier d'entraînement
TRAIN_DIR = "train"

# Taille à laquelle on redimensionne les images (224x224 est la taille standard pour AlexNet)
IMG_SIZE = 224

# Nombre d'images traitées en même temps pendant l'entraînement
BATCH_SIZE = 32

# Nombre de fois qu'on passe sur tout le dataset pendant l'entraînement
EPOCHS = 10

# Taux d'apprentissage : vitesse à laquelle le modèle ajuste ses poids
LR = 1e-4

# Graine aléatoire : pour que les résultats soient reproductibles
SEED = 42

# Nombre de processus parallèles pour charger les données
NUM_WORKERS = 2

# Les 3 classes de notre problème
LABELS = ["Health", "Rust", "Other"]

# Dictionnaire classe -> numéro (Health=0, Rust=1, Other=2)
LBL2ID = {k: i for i, k in enumerate(LABELS)}

# Dictionnaire numéro -> classe (inverse du précédent)
ID2LBL = {i: k for k, i in LBL2ID.items()}