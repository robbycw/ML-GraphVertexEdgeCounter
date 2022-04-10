from os.path import exists
from unittest import TestLoader
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from transforms import Rescale, RandomCrop, ToTensor
from VertexCountNet import VertexCountNet
from dataset import VertexGraphImageDatasets

# This file will train a neural network and save the model. 
# Based on a tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 

# Initialize transforms, training set, training data loader, and note classes go from 0-15 vertices. 

transform = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

batch_size = 4

trainset = VertexGraphImageDatasets(csv_file='./graphdata/verticesCount.csv', 
                                        root_dir='./graphdata/',
                                        transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
"""
# Asks the user if they want to train with the GPU... Not Implemented
gpu = input("Train with GPU? Enter 0 if so. ")
gpu = int(gpu)
if(gpu == 0):
    if(torch.cuda.is_available):
        # Have device set to cuda
        device = torch.device("cuda:0")
    else: 
        # No GPU available... 
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")
"""

# Asks the user if they want to load an existing Neural Network to train
# or to start training a new network. 
"""
loadNet = input("Train existing network or New Network? Existing=0 ; New=1 ")
loadNet = int(loadNet)
# Paths are assumed to start in the repository folder. 
if(loadNet == 0):
    # Loading Existing Network
    file = input("Enter path of existing neural network: ")
    path = "./" + file
    if(exists(path)):
        print("Path exists. Loading model from " + path)
    else: 
        print("Path " + path + " does not exist. Exiting program.")
        sys.exit()
    
    net = VertexCountNet(16)
    net.load_state_dict(torch.load(path))

elif(loadNet == 1):
    # Creating new network. 
    file = input("Enter name of the new neural network: ")
    path = "./" + file + ".pth"
    if(exists(path)):
        print("This neural network already exists. Please delete the file and try again. Exiting...")
        sys.exit()
    net = VertexCountNet(16)
    net = net.float()
else:
    print("Invalid input. Exiting...")
    sys.exit()
"""

# Send Net to proper device. Not Implemented
# net.to(device)

# Initialize neural network. 
net = VertexCountNet(16)
net = net.float()
path = "./model2.pth"

# Initialize Neural Network from given path, if desired. 
# net.load_state_dict(torch.load(path))

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):  # Number of times to loop over dataset 

    running_loss = 0.0

    for i, sample in enumerate(trainloader):
    
        # number of vertices in image; need to convert integer tensor to long tensor
        vert = sample['vertices']
        vert = vert.long()

        imag = sample['image']
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(imag)
        loss = criterion(outputs, vert)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.5f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(net.state_dict(), path)

# Model can be loaded with net.load_state_dict(torch.load(PATH))