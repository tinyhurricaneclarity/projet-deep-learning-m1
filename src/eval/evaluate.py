# evaluate.py
"""
Script d'évaluation unifié pour tous les modèles.
Génère des métriques et visualisations (matrice de confusion, F1 score)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score,
    recall_score, precision_score, classification_report
)

from src.config import BATCH_SIZE, PATH, get_class_names
from src.models import get_model, get_input_channels, get_input_size
from src.dataset.dataset_load import import_images, get_transforms, CustomImageDataset
from torch.utils.data import DataLoader


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Affiche et sauvegarde la matrice de confusion.
    
    Paramètres :
    - cm : matrice de confusion
    - class_names : noms des classes
    - save_path : chemin de sauvegarde
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardée : {save_path}")


def evaluate(model_name, Im_type, checkpoint_path=None, num_classes=None, 
             class_names=None, save_dir=None):
    """
    Évalue un modèle sur le jeu de test.

    Paramètres :
    - model_name : nom du modèle ('convnet', 'resnet18', 'resnet50', 'resnext50')
    - Im_type : type d'image ("RGB", "MS", "HS", "RGB_MS", ...)
    - checkpoint_path : chemin vers le modèle sauvegardé (None = auto)
    - num_classes : nombre de classes (None = auto-détecté)
    - class_names : liste des noms de classes (None = auto)
    - save_dir : dossier de sauvegarde (None = auto)
    
    Retour : (metrics, f1_par_classe, cm)
    """
    
    # Auto-détection des paramètres
    in_channels = get_input_channels(Im_type)
    input_size = get_input_size(Im_type)
    
    if num_classes is None:
        num_classes = 2 if "sans_other" in Im_type else 3
    
    if class_names is None:
        class_names = get_class_names(num_classes)
    
    if save_dir is None:
        from src.config import get_save_dir
        save_dir = get_save_dir(model_name, Im_type)
    
    if checkpoint_path is None:
        checkpoint_path = f"{save_dir}/best_{model_name}_{Im_type}_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Charger le split
    split_path = f"{save_dir}/split_{Im_type}.pth"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split non trouvé : {split_path}. Lancez d'abord l'entraînement.")
    
    dico_train_test = torch.load(split_path)
    print(f"Split chargé depuis {split_path}")

    # Importer les images de test
    Test = import_images(class_names, dico_train_test, "test")
    print(f"Test : {len(Test['images'])} images")

    # Transformations
    _, eval_transform = get_transforms(in_channels)

    # DataLoader
    test_dataset = CustomImageDataset(
        Test['images'], Test['labels'], transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Charger le modèle
    model = get_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=input_size
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Modèle chargé depuis {checkpoint_path}")

    # Évaluation
    print("\n=== Évaluation sur le jeu de test ===")
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images       = images.to(device)
            labels       = labels.squeeze()
            outputs      = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcul des métriques
    metrics = {
        "Accuracy":       accuracy_score(all_labels, all_preds),
        "Precision":      precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "Recall":         recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "F1-score macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    print("\n─── Résultats globaux ───")
    for name, score in metrics.items():
        print(f"  {name} : {score:.4f}")

    # F1-score par classe
    f1_par_classe = f1_score(all_labels, all_preds, average=None, zero_division=0)
    print("\n─── F1-score par classe ───")
    for i, classe in enumerate(class_names):
        print(f"  F1-score {classe} : {f1_par_classe[i]:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("\n─── Matrice de confusion ───")
    print(cm)

    # Rapport de classification détaillé
    print("\n─── Rapport de classification ───")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Sauvegarde de la matrice de confusion
    cm_path = f"{save_dir}/confusion_matrix_{model_name}_{Im_type}.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # Sauvegarde des métriques en fichier texte
    metrics_path = f"{save_dir}/evaluation_metrics_{model_name}_{Im_type}.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"=== Évaluation {model_name} - {Im_type} ===\n\n")
        f.write("Résultats globaux :\n")
        for name, score in metrics.items():
            f.write(f"  {name} : {score:.4f}\n")
        f.write("\nF1-score par classe :\n")
        for i, classe in enumerate(class_names):
            f.write(f"  {classe} : {f1_par_classe[i]:.4f}\n")
        f.write("\nMatrice de confusion :\n")
        f.write(str(cm))
        f.write("\n\nRapport de classification :\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(f"\nMétriques sauvegardées : {metrics_path}")
    print(f"Évaluation terminée. Résultats dans {save_dir}/")

    return metrics, f1_par_classe, cm


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluer un modèle")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["convnet", "resnet18", "resnet50", "resnext50"])
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    evaluate(
        model_name=args.model,
        Im_type=args.modality,
        checkpoint_path=args.checkpoint
    )