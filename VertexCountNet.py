import torch
import numpy as np
from torchvision import transforms, utils
import torch.nn.functional as F

# This convolutional neural network was based on the model given in the following tutorial: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html 

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

class VertexCountNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(VertexCountNet, self).__init__()

        # Net will take a 3 x 224 x 224 image of a graph

        # 1st Convolutional Layer: Takes 3 input channels (color), 6 output (features), 9x9 square convolution
        #   produces 6x216x216 output tensor. 
        #   max pooling by 2 reduces this to 6x108x108
        # 2nd Convolutional Layer: Takes 6 input channels (features), 10 output (features), 5x5 square convolution
        #   produces 10x104x104 output tensor. 
        #   max pooling by 2 reduces this to 10x52x52
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(9, 9))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(32 * 52 * 52, 500)
        self.fc2 = torch.nn.Linear(500, 100)
        self.fc3 = torch.nn.Linear(100, num_classes)
        self.sm = torch.nn.Softmax()

    def forward(self, x):
        # Max Pooling over 2x2 window (merges groups of 2x2 entries in Tensor)
        out = x.to(torch.float32)
        out = F.max_pool2d(F.relu(self.conv1(out)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = self.dropout1(out)
        out = out.view(-1, self.num_flat_features(out)) # reshapes Tensor from Convolutional shape to Linear shape
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.sm(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

