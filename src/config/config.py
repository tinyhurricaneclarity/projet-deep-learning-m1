# src/config.py
"""
Configuration et hyperparamètres pour le projet de détection de rouille jaune
"""

import os
from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================
PATH = "/home/mona/Documents/Projet/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train"
RESULTS_ROOT = Path(__file__).parent.parent.parent / "results"
CHECKPOINTS_ROOT = Path(__file__).parent.parent.parent / "checkpoints"

# Chemin par défaut des données d'entraînement
PATH = str(DATA_ROOT / "train")

# ============================================================================
# CLASSES
# ============================================================================

CLASS_NAMES = ["Health", "Rust", "Other"]
CLASS_NAMES_2 = ["Health", "Rust"]

# ============================================================================
# DONNÉES
# ============================================================================

NUM_DATA = 600  # 200 images par classe (3 classes)
NUM_DATA_2CLASSES = 400  # 200 images par classe (2 classes)

# Split des données
TRAIN_RATIO = 0.67  # 402 images pour train (134 par classe)
VAL_RATIO = 0.165   # 99 images pour val (33 par classe)
TEST_RATIO = 0.165  # 99 images pour test (33 par classe)

# ============================================================================
# MODALITÉS
# ============================================================================

MODALITES = {
    "RGB":           {"in_channels": 3,   "input_size": 64, "num_classes": 3},
    "MS":            {"in_channels": 5,   "input_size": 64, "num_classes": 3},
    "HS":            {"in_channels": 125, "input_size": 32, "num_classes": 3},
    "RGB_MS":        {"in_channels": 8,   "input_size": 64, "num_classes": 3},
    "MS_HS":         {"in_channels": 130, "input_size": 64, "num_classes": 3},
    "RGB_MS_HS":     {"in_channels": 133, "input_size": 64, "num_classes": 3},
    "MS_sans_other": {"in_channels": 5,   "input_size": 64, "num_classes": 2},
    "HS_sans_other": {"in_channels": 125, "input_size": 32, "num_classes": 2},
}

# ============================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT
# ============================================================================

BATCH_SIZE = 32
SEED_RANDOM = 42
SEED_TORCH = 42

# Early stopping
PATIENCE_EARLY_STOPPING = 10  # Nombre d'epochs sans amélioration avant arrêt

# Scheduler
PATIENCE_SCHEDULER = 5  # Pour ReduceLROnPlateau
FACTOR_SCHEDULER = 0.5  # Facteur de réduction du learning rate

# ============================================================================
# GRID SEARCH
# ============================================================================

GRID_PARAMS = {
    "num_epochs": [50, 100],
    "learning_rate": [0.001, 0.0001],
    "optimizer": ["Adam", "SGD"],
    "scheduler": ["StepLR", "ReduceLROnPlateau"],
}

# Configuration alternative pour tests rapides
GRID_PARAMS_FAST = {
    "num_epochs": [20],
    "learning_rate": [0.001],
    "optimizer": ["Adam"],
    "scheduler": ["ReduceLROnPlateau"],
}

# ============================================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================================

DEFAULT_CONFIG = {
    # Data
    'batch_size': BATCH_SIZE,
    'num_workers': 4,
    
    # Training
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'patience': PATIENCE_EARLY_STOPPING,
    
    # Model
    'num_classes': 3,
    
    # Device
    'device': 'cuda',
    'seed': SEED_TORCH,
}

# ============================================================================
# CONFIGURATIONS SPÉCIFIQUES PAR MODÈLE
# ============================================================================

MODEL_CONFIGS = {
    'convnet': {
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 50,
    },
    'resnet18': {
        'lr': 0.0001,
        'batch_size': 32,
        'epochs': 100,
    },
    'resnet50': {
        'lr': 0.0001,
        'batch_size': 16,
        'epochs': 100,
    },
    'resnext50': {
        'lr': 0.0001,
        'batch_size': 16,
        'epochs': 100,
    }
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_config(model_name, modality, **kwargs):
    """
    Crée une configuration pour un modèle et une modalité donnés
    
    Args:
        model_name: nom du modèle ('convnet', 'resnet18', 'resnet50', 'resnext50')
        modality: modalité d'image ('RGB', 'MS', 'HS', etc.)
        **kwargs: paramètres personnalisés supplémentaires
    
    Returns:
        dict: configuration complète
    """
    config = DEFAULT_CONFIG.copy()
    
    # Mise à jour avec la config du modèle
    if model_name in MODEL_CONFIGS:
        config.update(MODEL_CONFIGS[model_name])
    
    # Mise à jour avec les paramètres de la modalité
    if modality in MODALITES:
        config.update(MODALITES[modality])
    
    # Mise à jour avec les paramètres personnalisés
    config['model_name'] = model_name
    config['modality'] = modality
    config.update(kwargs)
    
    return config


def get_save_dir(model_name, modality):
    """Retourne le dossier de sauvegarde pour un modèle et une modalité"""
    save_dir = RESULTS_ROOT / model_name / modality
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def get_class_names(num_classes):
    """Retourne les noms de classes selon le nombre de classes"""
    return CLASS_NAMES_2 if num_classes == 2 else CLASS_NAMES
