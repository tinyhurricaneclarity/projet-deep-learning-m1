# Projet Deep Learning - Détection de la Rouille Jaune

Projet de classification d'images agricoles pour détecter la rouille jaune sur des cultures à l'aide de différentes modalités d'imagerie (RGB, multispectral, hyperspectral).

##  Structure du projet

```
projet-deep-learning-m1/
├── legacy #ancien code non unifié
├── data/
│   └── beyond-visible-spectrum-ai-for-agriculture-2026/
│       └── Kaggle Prepared/
│           ├── train/
│           │   ├── HS/
│           │   ├── MS/
│           │   └── RGB/
│           └── val/
│               ├── HS/
│               ├── MS/
│               └── RGB/
│
└── src
    ├── config
    │   ├── config.py       #hyperparamètres 
    │   └── __init__.py
    ├── dataset
    │   ├── dataset_load.py #chargement des données et transformations
    │   └── __init__.py
    ├── eval
    │   ├── evaluate.py 
    │   └── __init__.py
    ├── main.py             #appelle les fichiers nécéssaires pour faire tourner le modèle         
    ├── models
    │   ├── __init__.py
    │   └── models.py       #un fichier contenant tous les modèles : Convnet, Resnet18, Resnet50 et ResneXt50
    ├── requirements.txt
    ├── results
    │   ├── convnet/
    │   ├── resnet18/
    │   ├── resnet50/
    │   └── resnext50/
    └── train
        └── train.py 

```

##  Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

##  Usage

### Entraînement d'un modèle

```bash
# Entraîner ResNet18 sur RGB avec grid search
python main.py --model resnet18 --modality RGB --mode train

# Entraîner ResNet50 sur RGB+MS+HS sans grid search (plus rapide)
python main.py --model resnet50 --modality RGB_MS_HS --mode train --no-grid-search

# Entraîner ConvNet sur MS
python main.py --model convnet --modality MS --mode train
```

### Évaluation d'un modèle

```bash
# Évaluer le meilleur modèle ResNet18 RGB
python main.py --model resnet18 --modality RGB --mode eval

# Évaluer un checkpoint spécifique
python main.py --model resnet50 --modality HS --mode eval --checkpoint results/resnet50/HS/best_resnet50_HS_model.pth
```

### Entraînement et évaluation

```bash
# Tout en une seule commande
python main.py --model resnet18 --modality RGB_MS --mode both
```

##  Modalités disponibles

| Modalité       | Canaux | Taille | Classes | Description                     |
|----------------|--------|--------|---------|----------------------------------|
| RGB            | 3      | 64     | 3       | Images RGB classiques            |
| MS             | 5      | 64     | 3       | Multispectral                    |
| HS             | 125    | 32     | 3       | Hyperspectral                    |
| RGB_MS         | 8      | 64     | 3       | RGB + Multispectral              |
| MS_HS          | 130    | 64     | 3       | Multispectral + Hyperspectral    |
| RGB_MS_HS      | 133    | 64     | 3       | RGB + MS + HS (toutes modalités) |
| MS_sans_other  | 5      | 64     | 2       | MS sans classe "Other"           |
| HS_sans_other  | 125    | 32     | 2       | HS sans classe "Other"           |

##  Modèles disponibles

- **ConvNet** : Architecture custom from scratch
- **ResNet18** : ResNet avec 18 couches
- **ResNet50** : ResNet avec 50 couches (bottleneck blocks)
- **ResNeXt50** : ResNet avec grouped convolutions

##  Résultats

Les résultats de chaque entraînement sont sauvegardés dans `results/[model]/[modality]/` :

- `grid_search_results_[modality].csv` : Résultats du grid search
- `best_[model]_[modality]_model.pth` : Meilleur modèle
- `loss_[model]_[modality].png` : Courbe de loss
- `accuracy_[model]_[modality].png` : Courbe d'accuracy
- `confusion_matrix_[model]_[modality].png` : Matrice de confusion (après évaluation)

##  Configuration

Les hyperparamètres sont définis dans `src/config.py` :

- Grid search : `GRID_PARAMS`
- Batch size : `BATCH_SIZE`
- Early stopping : `PATIENCE_EARLY_STOPPING`
- Scheduler : `PATIENCE_SCHEDULER`, `FACTOR_SCHEDULER`

##  Classes

Le dataset contient 3 classes :
- **Health** : Plantes saines
- **Rust** : Rouille jaune
- **Other** : Autres maladies/états

Pour les modalités `sans_other`, seules les classes Health et Rust sont utilisées.

##  Grid Search

Le grid search teste automatiquement différentes combinaisons d'hyperparamètres :
- Nombre d'epochs : 50, 100
- Learning rate : 0.001, 0.0001
- Optimizer : Adam, SGD
- Scheduler : StepLR, ReduceLROnPlateau

Pour désactiver le grid search et utiliser les paramètres par défaut :
```bash
python main.py --model resnet18 --modality RGB --mode train --no-grid-search
```

