from unittest import TestLoader
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

transform = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

batch_size = 1

trainset = VertexGraphImageDatasets(csv_file='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/verticesCount.csv', 
                                        root_dir='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/',
                                        transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


net = VertexCountNet(16)
net = net.float()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # Number of times to loop over dataset 

    running_loss = 0.0

    for i, sample in enumerate(trainloader):
    
        # number of vertices in image; need to convert integer tensor to long tensor
        vert = sample['vertices']
        vert = vert.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(sample['image'])
        loss = criterion(outputs, vert)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
PATH = './vertCount_Net_1.pth'
torch.save(net.state_dict(), PATH)

# Model can be loaded with net.load_state_dict(torch.load(PATH))