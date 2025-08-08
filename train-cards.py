import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

#print('System Version:', sys.version)
#print('PyTorch version', torch.__version__)
#print('Torchvision version', torchvision.__version__)
#print('Numpy version', np.__version__)
#print('Pandas version', pd.__version__)

#datasets and dataloader
class PlayingCardDataset(Dataset):
    #tells class what to do when its created
    def __init__(self, data_dir, transform=None):
        self.data= ImageFolder(data_dir, transform=transform)

    #dataloader needs to know how many examples we have in a dataset
    def __len__(self):
        return len(self.data)
    
    #takes index location in dataset and returns an item
    def __getitem__(self, idx):
        return self.data[idx]
    
    #returns classes from image folder
    def classes(self):
        return self.data.classes

dataset= PlayingCardDataset(data_dir= 'train')
#testing len function
print(len(dataset)) #7624
#testing get item function 
print(dataset[5]) 

