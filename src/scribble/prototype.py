import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

#Hyper
latent_dim = 128
batch_size = 128
lr = 0.0002
epochs = 50
device = torch.device("cuda")


