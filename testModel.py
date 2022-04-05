from unittest import TestLoader
from bitarray import test
from os.path import exists
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from transforms import Rescale, RandomCrop, ToTensor
from VertexCountNet import VertexCountNet
from dataset import VertexGraphImageDatasets

# This file will test a given neural network's ability to count vertices.  

# Based on a tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 


transform = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])

batch_size = 1

testset = VertexGraphImageDatasets(csv_file='./testdata/verticesCount.csv', 
                                        root_dir='./testdata/',
                                        transform=transform)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

# Requests path to NN save file.
# Paths are assumed to start in the repository folder.
loadNet = input("Enter path to neural network save file: ")
path = "./" + loadNet
 
if(exists(path)):
    # Loading Existing Network
    print("Path exists. Loading model from " + path)
else: 
    print("Path " + path + " does not exist. Exiting program.")
    sys.exit()

# Loads the saved neural network model
net = VertexCountNet(16)
net.load_state_dict(torch.load(path))

# Tests the accuracy of the whole test data set with saved VertexCountNet
correct = 0
total = 0

with torch.no_grad(): # not needed since gradients are only needed when training
    for data in testloader:

        image = data['image']
        vertices = data['vertices']

        outputs = net(image)

        _, predicted = torch.max(outputs.data, 1)
        total += vertices.size(0)
        correct += (predicted == vertices).sum().item()

# Print results
print(f'Correct:  {correct}')
print(f'Total Images:  {total}')
print(f'Accuracy of the neural network on the test images: {100 * correct // total} %')
print('Finished Testing')
