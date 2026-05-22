# main.py
"""
Point d'entrée principal du projet de détection de rouille jaune
"""

import argparse
import torch
from src.config.config import PATH, MODALITES, get_save_dir, get_class_names
from src.train.train import train
from src.eval.eval import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Détection de la rouille jaune sur images agricoles"
    )
    
    # Arguments principaux
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["convnet", "resnet18", "resnet50", "resnext50"],
        help="Modèle à utiliser"
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=list(MODALITES.keys()),
        help="Modalité d'imagerie à utiliser"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "both"],
        help="Mode : train, eval, ou both"
    )
    
    # Options d'entraînement
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Désactiver le grid search et utiliser les paramètres par défaut"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=PATH,
        help="Chemin vers les données d'entraînement"
    )
    
    # Options d'évaluation
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers le checkpoint pour l'évaluation"
    )
    
    args = parser.parse_args()
    
    # Récupération des paramètres de la modalité
    modality_config = MODALITES[args.modality]
    num_classes = modality_config["num_classes"]
    class_names = get_class_names(num_classes)
    
    save_dir = get_save_dir(args.model, args.modality)
    
    print(f"\n{'='*60}")
    print(f"Modèle    : {args.model}")
    print(f"Modalité  : {args.modality}")
    print(f"Canaux    : {modality_config['in_channels']}")
    print(f"Taille    : {modality_config['input_size']}x{modality_config['input_size']}")
    print(f"Classes   : {num_classes} {class_names}")
    print(f"Save dir  : {save_dir}")
    print(f"{'='*60}\n")
    
    # MODE TRAIN
    if args.mode in ["train", "both"]:
        print(">>> Lancement de l'entraînement.")
        results = train(
            model_name=args.model,
            Im_type=args.modality,
            path=args.data_path,
            num_classes=num_classes,
            class_names=class_names,
            save_dir=save_dir,
            use_grid_search=not args.no_grid_search
        )
        print(f"\nEntraînement terminé. Résultats dans {save_dir}/")
    
    # MODE EVAL
    if args.mode in ["eval", "both"]:
        print("\n>>> Lancement de l'évaluation.")
        
        # Déterminer le checkpoint à utiliser
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = f"{save_dir}/best_{args.model}_{args.modality}_model.pth"
        
        print(f"Checkpoint : {checkpoint_path}")
        
        # Vérifier que le checkpoint existe
        import os
        if not os.path.exists(checkpoint_path):
            print(f"ERREUR : Le checkpoint {checkpoint_path} n'existe pas.")
            print("Lancez d'abord l'entraînement ou spécifiez un checkpoint avec --checkpoint")
            return
        
        
        metrics, f1_par_classe, cm = evaluate(
            model_name=args.model,
            Im_type=args.modality,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            class_names=class_names,
            save_dir=save_dir
        )
        
        print("Évaluation terminée.")
    
    print(f"\n{'='*60}")
    print(f"✓ Terminé. Résultats dans {save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
