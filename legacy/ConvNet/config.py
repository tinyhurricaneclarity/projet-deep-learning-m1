# config.py
# Fichier de configuration central pour le ConvNet
# Modifier ce fichier pour adapter le code à votre environnement

# PASS : mot de passe pour identifier l'utilisateur
PASS = "mypass"


# CHEMINS DES DONNÉES

paths = {
    "mypass": "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train",
    "profpass": "/path/to/data"  # à compléter par la prof
}

PATH = paths[PASS]


# CLASSES
CLASS_NAMES = ["Health", "Rust", "Other"]
NUM_CLASSES = len(CLASS_NAMES)


# HYPERPARAMÈTRES
BATCH_SIZE = 32
NUM_DATA   = 600  # nombre total d'images (200 par classe × 3 classes)
NUM_DATA_2CLASSES = 400  # 2 classes × 200

# Grid search : valeurs à tester
GRID_PARAMS = {
    "num_epochs":    [100, 200],
    "learning_rate": [0.001, 0.0001],
    "optimizer":     ["Adam", "SGD"],
    "scheduler":     ["ReduceLROnPlateau", "StepLR"],
}

# Early stopping
PATIENCE_EARLY_STOPPING = 15
PATIENCE_SCHEDULER      = 5
FACTOR_SCHEDULER        = 0.5

# Seeds pour la reproductibilité
SEED_RANDOM = 42
SEED_TORCH  = 42