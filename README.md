# Table des matières

[Projet](#Projet)
[Arboresence](#arborescence)

# Projet

Kaggle competition :
Beyond Visible Spectrum: AI for Agriculture 2026
Automated Multimodal Crop Disease Diagnosis from multimodal remote sensing imagery 5th

L'objetif est d'entraîner un modèle sur des images RGB, multispectrales et hyperspectrales provenant d'un champs, pour la reconnaissance de la rouille du blé. 
Les images sont prises par un drone : DJI M600 Pro UAV with an S185 snapshot hyperspectral sensor (UAV imagery), à 60m de hauteur. Résolution spectrale : 4cm/pixel.

# Arborescence

```
├── data/
│    beyond-visible-spectrum-ai-for-agriculture-2026
│    ├── Kaggle Prepared/ 
│        ├──train
│            ├──HS
│            ├──MS
│            ├──RGB
│        ├──val
│            ├──HS
│            ├──MS
│            ├──RGB  
│  
├── src/
│     ├── config/
│       ├── config.py #hyperparamètres + chemins
│
│     ├── dataset/
│       ├── dataset_load.py #Load, data augmentation
│
│     ├── models/
│       ├── __init__.py
│       ├── convnet.py
│       ├── resnet.py
│       └── resnext.py
│
├── train.py                    # Tout-en-un : training + losses + metrics
├── evaluate.py                 # Évaluation et visualisation
├── main.py
│
│
│      ├── results/                       # Dossier pour sauvegarder les résultats
│         ├── convnet/
│         ├── resnet18/
│         ├── resnet50/
│         └── resnext50/
│      
├── requirements.txt
├── README.md
└── .gitignore
```
# Matériel 

Linux saruman 6.12.74+deb12-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.74-2~bpo12+1 (2026-03-13) x86_64 GNU/Linux

Architecture :                              x86_64
  Mode(s) opératoire(s) des processeurs :   32-bit, 64-bit
  Tailles des adresses:                     46 bits physical, 48 bits virtual
  Boutisme :                                Little Endian
Processeur(s) :                             24
  Liste de processeur(s) en ligne :         0-23
Identifiant constructeur :                  GenuineIntel
  Nom de modèle :                           12th Gen Intel(R) Core(TM) i9-12900
    Famille de processeur :                 6
    Modèle :                                151
    Thread(s) par cœur :                    2
    Cœur(s) par socket :                    16
    Socket(s) :                             1
    Révision :                              2

2 GPUS
Principal : NVIDIA GeForce RTX 3060
Intégré au processeur (iGPU) : Intel UHD Graphics 770


# Protocole expérimental


base commune pour chaque modèle

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',        # réduit quand la val loss ne s'améliore plus
    factor=0.5,        # divise le learning rate par 2
    patience=5,        # attend 5 epochs sans amélioration avant de réduire
)

batch_size    = 32
num_epochs    = 100
learning_rate = 0.001



avec - sans dataaugmentation


K fold ?
Si ton dataset était très petit (< 1000 exemples) et que chaque exemple compte
Si tu voulais faire du model ensembling (moyenner les prédictions des k modèles)
Si tu avais un déséquilibre de classes important à gérer
Pour une compétition Kaggle avec ResNet18, garde ton split actuel. Le K-Fold est une optimisation avancée qui ne vaut le coût que si tu cherches à gratter les derniers points sur le leaderboard