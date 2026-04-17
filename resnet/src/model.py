#Packages
import torch
import torch.nn as nn

#Premier bloc résiduel simple (pas resnet18) pour comprendre
class ResidualBlock(nn.Module): #hérite de module
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU(inplace=True), use_batchnorm=True):
        super(ResidualBlock, self).__init__()
         #données des couches de convolution :
         # canaux entrée et sortie
         # taille de kernel par défaut 3 on peut le modifier,
         # "acitvation" est la fonction d'activation. ici reLU (il y en a d'autres) empêche le problème du vanishing gradient, surtout pour les réseaux profonds.
         # batch_norm : permet de normaliser les données sortant de la couche de convolution pour éviter d'avoir des données trop dispersées

        padding = kernel_size // 2 #pour garder les dimensions constantes

        #première couche de convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # deuxième couche convolution (meme chose que premiere)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        #définition de la fonction d'activation
        self.activation = activation

        #pour vérifier qu'il n'y a pas de problèmes
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1en , bias=False)
        else:
            self.shortcut = nn.Identity()

    #Faire passer toutes les données jusqu'à la fin (voir fonctionnement resnet)
    def forward(self, x):
        shortcut = self.shortcut(x) #shortcut est la donnée à la fin du bloc résiduel

        #on fait passer nos données à travers chaque couche
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut #ajout des données de base (shortcut) à nos données passée dans les couches de convolution
        out = self.activation(out)
        return out
