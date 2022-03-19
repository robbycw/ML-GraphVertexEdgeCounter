# The objective of this file is to write the code for taking the Sage graph images
# and using PyTorch classes and transforms to interpret their data and prepare
# the data for machine learning. 

# I used the following tutorial by Sasank Chilamkurthy: 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from transforms import Rescale, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class VertexGraphImageDatasets(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: (string) Contains path to csv file storing image names + # of vertices
        root_dir: (string) Path to directory with all images
        transform: (callable, optional) Optional transform for images. 
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.csv.iloc[idx, 0])
        image = io.imread(img_name)
        vertices = self.csv.iloc[idx, 1]
        vertices = vertices.astype(int)
        sample = {'image': image, 'vertices': vertices}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
# Short test that shows the dataset is able to access the .csv file and print each image array and vertex count.
vert_dataset = VertexGraphImageDatasets(csv_file='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/verticesCount.csv', 
                                        root_dir='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/')

for i in range(len(vert_dataset)):
    sample = vert_dataset[i]
    print(i, sample['image'].shape, sample['vertices'])

# This following test does the same thing, but applies transforms to the images and makes sure that the image sizes were changed. 
transformed_dataset = VertexGraphImageDatasets(csv_file='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/verticesCount.csv', 
                                        root_dir='C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/',
                                        transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['vertices'])

    if i == 5: # We only need to see the first few to confirm the transforms are working. 
        break

# This tests applies transforms and iterates over the dataset with a DataLoader. 

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['vertices'])

    if i_batch == 3: 
        break
"""