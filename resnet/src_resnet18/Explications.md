# Sauvegarde des modèles

Qu'est ce qu'un state_dict ?
docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
Exemples :
poids des convolutions
biais
batchnorm,