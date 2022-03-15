import torch
import numpy as np
from torchvision import transforms, utils
import torch.functional as F

# This convolutional neural network was based on the model given in the following tutorial: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html 
class VertexCountNet(torch.nn.Module):

    def __init__(self):
        super(VertexCountNet, self).__init__()

        # Net will take a 3 x 224 x 224 image of a graph

        # 1st Convolutional Layer: Takes 3 input channels (color), 6 output (features), 9x9 square convolution
        #   produces 6x216x216 output tensor. 
        #   max pooling by 2 reduces this to 6x108x108
        # 2nd Convolutional Layer: Takes 6 input channels (color), 16 output (features), 5x5 square convolution
        #   produces 10x104x104 output tensor. 
        #   max pooling by 4 reduces this to 10x26x26
        self.conv1 = torch.nn.Conv2d(3, 6, 9)
        self.conv2 = torch.nn.Conv2d(6, 10, 4)

        self.fc1 = torch.nn.Linear(10 * 26 * 26, 500)
        self.fc2 = torch.nn.Linear(500, 100)
        self.fc3 = torch.nn.Linear(100, 10)

    def forward(self, x):
        # Max Pooling over 2x2 window (merges groups of 2x2 entries in Tensor)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2(F.relu(self.conv2(x)), 4)
        x = x.view(-1, self.num_flat_features(x)) # reshapes Tensor from Convolutional shape to Linear shape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

