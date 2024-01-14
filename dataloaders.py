from torch.utils.data import random_split
from build_dataset import CustomGraphDataset
from torch_geometric.data import DataLoader
from graph_image_builder import graph_pic_gen
import torch.nn as nn 
import torch

def return_embeddings(x):
    im_lin1 = nn.Linear(160000, 30, bias=False)
    # Embed each image (left or right)
    embs = []
    x  = x
    # x = torch.from_numpy(x).float()
    # Assuming h has a size of [400 x 400], you may want to flatten it
    h = x.view(-1)
    # Apply the linear layer
    h_i = im_lin1(h)
    
    # Reshape the result to [batch_size x 1 x 1 x embedding_size]
    h_i = h_i.view(-1, 1, 1, 30)
    
    # Append to the list of embeddings
    embs.append(h_i)
    
    # Concatenate the embeddings along the fourth dimension
    h = torch.cat(embs, dim=2)

    return h

def create_data_loaders(batch_size, train_split=0.8):
    dataset = CustomGraphDataset()
    dataset = []
    doubleset  = []
    for i in range(3000):
        graph, image = graph_pic_gen()
        pair = (graph, image)
        dataset.append(graph)
        doubleset.append(pair)


    # Adjust split sizes based on your dataset
    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total dataset size: {len(dataset)}")
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {valid_size}")
    print("======")
    
    return pair, valid_loader, dataset

train_loader, valid_loader, dataset = create_data_loaders(1)
