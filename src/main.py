from models.convnet import ConvNet
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50
from models.resnext50 import ResNeXt50

import argparse
from config.config import PATH, CLASS_NAMES
from train.train_convnet import train
from eval import evaluate

MODELS = {
    "convnet": ConvNet,
    "resnet18": ResNet18,
    "resnet50": ResNet50,
    "resnext50": ResNeXt50,
}

MODALITES = {
    "RGB":           (3,   64, 3),
    "MS":            (5,   64, 3),
    "HS":            (125, 32, 3),
    "RGB_MS":        (8,   64, 3),
    "MS_HS":         (130, 64, 3),
    "RGB_MS_HS":     (133, 64, 3),
    "MS_sans_other": (5,   64, 2),
    "HS_sans_other": (125, 32, 2),
}

def main():
    parser = argparse.ArgumentParser(description="Détection rouille jaune")
    parser.add_argument(
        "--modalite",
        type=str,
        required=True,
        choices=list(MODALITES.keys()),
        help="Modalité d'imagerie à utiliser"
    )

    parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=list(MODELS.keys()),
    help="Modèle à utiliser"
    )

    args = parser.parse_args()
    model_name = args.model
    Im_type = args.modalite

    in_channels, input_size, num_classes = MODALITES[Im_type]

    model_class = MODELS[model_name]
    model = model_class(
    in_channels=in_channels,
    num_classes=num_classes
    )

    # Adapter CLASS_NAMES selon num_classes
    class_names = ["Health", "Rust"] if num_classes == 2 else CLASS_NAMES

    save_dir = f"saved_models/{model_name}/{Im_type}"

    print(f"\n{'='*60}")
    print(f"Modalité : {Im_type}")
    print(f"Canaux : {in_channels} | Taille : {input_size}x{input_size} | Classes : {num_classes}")
    print(f"{'='*60}\n")

    print(">>> Lancement de l'entraînement.")
    results = train(
        model=model,
        Im_type=Im_type,
        path=PATH,
        in_channels=in_channels,
        input_size=input_size,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir
    )

    print("\n>>> Lancement de l'évaluation.")
    metrics, f1_par_classe, cm = evaluate(
        model=model,
        Im_type=Im_type,
        in_channels=in_channels,
        input_size=input_size,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir
    )

    print(f"\n{'='*60}")
    print(f"Terminé. Résultats sauvegardés dans {save_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()