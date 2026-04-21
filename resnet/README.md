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
│   ├── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```