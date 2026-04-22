# Sauvegarde des modèles

Qu'est ce qu'un state_dict ?
docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
Exemples :
poids des convolutions
biais
batchnorm

torch.save https://docs.pytorch.org/docs/stable/generated/torch.save.html
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)
Saves an object to a disk file.

# Chargement des modèles

Quand on a sauvegardé un modèle, pour pouvoir l'utiliser il faut le charger
voir fichier eval.py section chargement du modèle à tester
model = ton_modele()

# Comment faire pour avoir les memes train, val et test set pour pouvoir les travailler dans deux fichiers différents train.py et eval.py (pas besoin si on lance dans le meme fichier)
Soit on garde le meme seed dans le create_dataloader (ici 42). Mais risqué si on a changé quelque chose entre les deux fichiers.
Soit on sauvegarde les indices des images de train, val et test (générés dans train.py), puis on les charges dans eval.py. 
Attention : bien faire la différence entre create_dataloader (qui fait un split random (ou à partir d'un seed), et la fonction DataLoader de pytorch, qui crée simplement les loaders)

# Rappel des loaders 
Un DataLoader permet de découper le dataset en batch. Shuffle si activé