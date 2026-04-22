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
resnet/
│
├── data/
│    data/beyond-visible-spectrum-ai-for-agriculture-2026
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
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_load.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├──   train.py
│
├── results/
│   ├── figures/
│   ├── saved_models/
    ├── test_indices.pth <- indices des train, val et test pour garantir que les datasets sont les memes entre train.py et eval.py
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

# Ce qu'il reste à faire

FAIT sauvegarde des best val loss et acc 
FAIT faire test
faire sauvegarde des résultats de test. affichage graphique de la matrice de confusion ?
Faire sauvegarde sous forme de tableau des métriques

data load en prenant au hasard parmi les classes
FAIT F1 score et confusion matrix. Reste générer les métriques pour comparer les modèles. 
Faire autres métriques : rappel, sensibilité, précision. intéret ?

grid search

Résumer les résultats sur overleaf

K fold ?