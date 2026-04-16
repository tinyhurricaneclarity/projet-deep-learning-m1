import os #permet d'utiliser des fonctionnalités du système d'exploitation (mkdir, lecture...)
import itertools
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd
from pathlib import Path
import skimage as ski

from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim