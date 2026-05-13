# eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score,
    recall_score, precision_score
)
from config import CLASS_NAMES, BATCH_SIZE, PATH
from model import ConvNet
from data_load import import_images, get_transforms, CustomImageDataset
from torch.utils.data import DataLoader


def evaluate(Im_type, in_channels, input_size, num_classes=3, class_names=None, save_dir="saved_models"):
    """
    Evalue le meilleur modèle ConvNet sur le jeu de test.

    Paramètres :
    - Im_type     : type d'image ("RGB", "MS", "HS", ...)
    - in_channels : nombre de canaux
    - input_size  : taille spatiale des images
    - num_classes : nombre de classes (3 par défaut)
    - class_names : liste des noms de classes (None = CLASS_NAMES complet)
    - save_dir    : dossier contenant le modèle sauvegardé
    """

    if class_names is None:
        class_names = CLASS_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    split_path = f"{save_dir}/split_{Im_type}.pth"
    dico_train_test = torch.load(split_path)
    print(f"Split chargé depuis {split_path}")

    Test = import_images(class_names, dico_train_test, "test")

    _, eval_transform = get_transforms(in_channels)

    test_dataset = CustomImageDataset(
        Test['images'], Test['labels'], transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet(
        in_channels=in_channels,
        input_size=input_size,
        num_classes=num_classes
    ).to(device)

    model_path = f"{save_dir}/best_{Im_type}_global.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Modèle chargé depuis {model_path}")

    print("\nEvaluation sur le jeu de test.")
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

    metrics = {
        "Accuracy":       accuracy_score(all_labels, all_preds),
        "Precision":      precision_score(all_labels, all_preds, average="macro"),
        "Recall":         recall_score(all_labels, all_preds, average="macro"),
        "F1-score macro": f1_score(all_labels, all_preds, average="macro"),
    }

    print("\n─── Résultats sur le jeu de test ───")
    for name, score in metrics.items():
        print(f"  {name} : {score:.4f}")

    f1_par_classe = f1_score(all_labels, all_preds, average=None)
    print("\n─── F1-score par classe ───")
    for i, classe in enumerate(class_names):
        print(f"  F1-score {classe} : {f1_par_classe[i]:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\n─── Matrice de confusion ───")
    print(cm)

    return metrics, f1_par_classe, cm