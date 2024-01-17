import os
import torch
import random
from torch_geometric.data import Dataset
from graph_image_builder import graph_pic_gen

class CustomGraphDataset():
    def __init__(self, size = 10):
        super(CustomGraphDataset, self).__init__()
        self.size = size
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['data.pt']

    def len(self):
        return len(self.data)

    def __len__(self):
         return len(self.data)
    
    def get(self, idx):
        return self.data[idx]

    def process(self):
        # Check if the dataset is already processed
            self.data = []
            for i in range(self.size):
                graph_data, image_data = graph_pic_gen()
                self.data.append((graph_data, image_data))

    def __getitem__(self, idx):
        data = self.data[idx]

        return data

