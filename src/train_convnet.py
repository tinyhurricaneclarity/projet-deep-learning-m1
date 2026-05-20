# train.py
import os
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from config import (
    CLASS_NAMES, BATCH_SIZE, NUM_DATA, NUM_DATA_2CLASSES, GRID_PARAMS,
    PATIENCE_EARLY_STOPPING, PATIENCE_SCHEDULER, FACTOR_SCHEDULER,
    SEED_RANDOM, SEED_TORCH
)
from models import convet
from dataset.dataset_load import alea_train_test, sufix_and_path, import_images, get_transforms, create_dataloaders


def train(Im_type, path, in_channels, input_size, num_classes=3, class_names=None, save_dir="saved_models"):
    """
    Boucle d'entraînement générique avec grid search.

    Paramètres :
    - Im_type     : type d'image ("RGB", "MS", "HS", "RGB_MS", ...)
    - path        : chemin vers les données
    - in_channels : nombre de canaux (3=RGB, 5=MS, 125=HS, ...)
    - input_size  : taille spatiale des images (64 ou 32 pour HS)
    - num_classes : nombre de classes (3 par défaut)
    - class_names : liste des noms de classes (None = CLASS_NAMES complet)
    - save_dir    : dossier de sauvegarde des modèles
    """

    if class_names is None:
        class_names = CLASS_NAMES

    num_data = NUM_DATA_2CLASSES if num_classes == 2 else NUM_DATA

    import random
    random.seed(SEED_RANDOM)
    torch.manual_seed(SEED_TORCH)
    torch.cuda.manual_seed_all(SEED_TORCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Split des données
    n_val  = 66 if num_classes == 2 else 99
    n_test = 66 if num_classes == 2 else 99
    dico_train_test = alea_train_test(num_data, class_names, n_val=n_val, n_test=n_test)
    sufix_and_path(Im_type, dico_train_test, path)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(dico_train_test, f"{save_dir}/split_{Im_type}.pth")

    # Chargement des images
    Train = import_images(class_names, dico_train_test, "train")
    Val   = import_images(class_names, dico_train_test, "val")
    Test  = import_images(class_names, dico_train_test, "test")

    print(f"Train : {len(Train['images'])} images")
    print(f"Val   : {len(Val['images'])} images")
    print(f"Test  : {len(Test['images'])} images")

    train_transform, eval_transform = get_transforms(in_channels)

    train_loader, val_loader, test_loader = create_dataloaders(
        Train, Val, Test, train_transform, eval_transform
    )

    criterion = nn.CrossEntropyLoss()

    results_summary = []
    config_counter  = 0
    best_f1_global  = 0
    best_train_losses     = []
    best_val_losses       = []
    best_train_accuracies = []
    best_val_accuracies   = []
    best_config_str       = ""

    # GRID SEARCH
    for (num_epochs, learning_rate, optimizer_name, scheduler_name) in itertools.product(
        GRID_PARAMS["num_epochs"],
        GRID_PARAMS["learning_rate"],
        GRID_PARAMS["optimizer"],
        GRID_PARAMS["scheduler"],
    ):
        config_str = (
            f"config{config_counter}_"
            f"epochs{num_epochs}_"
            f"lr{learning_rate}_"
            f"opt{optimizer_name}_"
            f"sched{scheduler_name}"
        )
        print(f"\n=== {config_str} ===")
        config_counter += 1

        model = ConvNet(
            in_channels=in_channels,
            input_size=input_size,
            num_classes=num_classes
        ).to(device)

        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        if scheduler_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=FACTOR_SCHEDULER, patience=PATIENCE_SCHEDULER
            )

        train_losses, val_losses         = [], []
        train_accuracies, val_accuracies = [], []
        best_val_loss            = float('inf')
        epochs_sans_amelioration = 0
        stop_training            = False

        # BOUCLE D'ENTRAÎNEMENT
        for epoch in range(1, num_epochs + 1):

            if stop_training:
                continue

            start_time = time.time()

            model.train()
            running_loss  = 0.0
            correct_train = 0
            total_train   = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze()

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss  += loss.item() * images.size(0)
                _, predicted   = torch.max(outputs, 1)
                total_train   += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc  = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            model.eval()
            running_val_loss = 0.0
            correct_val      = 0
            total_val        = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels         = labels.squeeze()
                    outputs        = model(images)
                    loss_val       = criterion(outputs, labels)

                    running_val_loss += loss_val.item() * images.size(0)
                    _, predicted      = torch.max(outputs, 1)
                    total_val        += labels.size(0)
                    correct_val      += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader.dataset)
            val_acc  = correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            epoch_time = time.time() - start_time

            print(f"Epoch [{epoch:02d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Temps: {epoch_time:.2f}s")

            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss            = val_loss
                epochs_sans_amelioration = 0
                torch.save(model.state_dict(), f"{save_dir}/best_{Im_type}_{config_str}.pth")
                print(f"  -> Meilleur modèle sauvegardé (Val Loss: {val_loss:.4f})")
            else:
                epochs_sans_amelioration += 1
                if epochs_sans_amelioration >= PATIENCE_EARLY_STOPPING:
                    print(f"Early stopping à l'epoch {epoch}")
                    stop_training = True

        # EVALUATION SUR LE JEU DE TEST
        model.load_state_dict(torch.load(f"{save_dir}/best_{Im_type}_{config_str}.pth"))
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

        f1_macro      = f1_score(all_labels, all_preds, average="macro")
        f1_par_classe = f1_score(all_labels, all_preds, average=None)

        print(f"F1-score macro (test) : {f1_macro:.4f}")

        if f1_macro > best_f1_global:
            best_f1_global        = f1_macro
            best_train_losses     = train_losses.copy()
            best_val_losses       = val_losses.copy()
            best_train_accuracies = train_accuracies.copy()
            best_val_accuracies   = val_accuracies.copy()
            best_config_str       = config_str
            torch.save(model.state_dict(), f"{save_dir}/best_{Im_type}_global.pth")
            print(f"  -> Meilleur modèle global sauvegardé (F1: {f1_macro:.4f})")

        results_summary.append({
            "config":        config_str,
            "modalite":      Im_type,
            "num_epochs":    num_epochs,
            "learning_rate": learning_rate,
            "optimizer":     optimizer_name,
            "scheduler":     scheduler_name,
            "val_loss":      best_val_loss,
            "f1_macro":      f1_macro,
            "f1_Health":     f1_par_classe[0],
            "f1_Rust":       f1_par_classe[1],
            "f1_Other":      f1_par_classe[2] if num_classes == 3 else None,
        })

    # COURBES DE LA MEILLEURE CONFIGURATION
    epochs_range = range(1, len(best_train_losses) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_range, best_train_losses, label='Train Loss')
    plt.plot(epochs_range, best_val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Courbe de loss - ConvNet {Im_type} ({best_config_str})")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_convnet_{Im_type}.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_range, best_train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, best_val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Courbe d'accuracy - ConvNet {Im_type} ({best_config_str})")
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy_convnet_{Im_type}.png")
    plt.close()

    print(f"\nCourbes sauvegardées dans {save_dir}/")

    df = pd.DataFrame(results_summary)
    df.to_csv(f"{save_dir}/grid_search_results_{Im_type}.csv", index=False)
    print(f"Résultats sauvegardés dans grid_search_results_{Im_type}.csv")

    print("\n─── Top 5 configurations ───")
    for r in sorted(results_summary, key=lambda x: x["f1_macro"], reverse=True)[:5]:
        print(f"  {r['config']} → F1={r['f1_macro']:.4f} | Val Loss={r['val_loss']:.4f}")

    return results_summary