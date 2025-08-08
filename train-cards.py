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

#datasets 
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
print(dataset[1]) 
#displayes image with label
image, label= dataset[10]
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

#get dictionary associating target values with folder names
data_dir = 'train'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

#make sure images are the same size
transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#prints tensor(data structure used to store data) of an image 
data_dir='train'
dataset= PlayingCardDataset(data_dir, transform)
print(dataset[100])

#iterate over dataset:
#for image, label in dataset:

#dataloader
#random data pull of size 32 (batch size must be in power of twos (16, 32, 64,...))
#no need to shuffle on test or validation set
dataloader= DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader: 
    break

print(images.shape, labels.shape)

#pytorch model
class SimpleCardClassifier(nn.Module):
    #define difference parts of the model
    def __init__(self, num_classes= 53):
        super(SimpleCardClassifier, self).__init__()

        self.base_model= timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size=1280
        #making a classifier
        self.classifier= nn.Linear(enet_out_size, num_classes)
    #take in an example or batch to connect them and return output
    def forward(self,x):
        x = self.features(x)
        output = self.classifier(x)
        return output
#prints model
model = SimpleCardClassifier(num_classes=53)
print(str(model)[:500])
#tests to see if model accepts input data we provide
#prints out tensor info
print(model(images))

