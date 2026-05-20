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
projet-deep-kearning-m1/
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
├── .gitignore
├── legacy                      #dossier pour les anciens codes
```
# Matériel 

Ordinateur du deuxième étage du CREMI :

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
