from dgl.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import dgl
import torch
import dgl.data
import matplotlib.pyplot as plt
import cv2
import random
from torch_geometric.data import Data
from torchvision.utils import save_image
from torchvision import transforms

def graph_pic_gen():
    templates = [[0, 1], [2, 0], [2, 1], [2, 2]]
    images = [[r".\img\fles.png", r".\img\lama.png", r".\img\tulp.png"], [r".\img\pilaar.png", r".\img\tafel.png", r".\img\stoel.png"], [r".\img\schilderij.png", r".\img\leeg.png", r".\img\leeg.png"]]

    seed = random.randint(0, 3)
    seed2 = random.randint(0, 3)

    images_list = [templates[seed][0], templates[seed][1], templates[seed2][0], templates[seed2][1]]
    x = [[x] for x in images_list]
    images_list = [images[images_list[0]][random.randint(0, 2)], images[images_list[1]][random.randint(0, 2)], images[images_list[2]][random.randint(0, 2)], images[images_list[3]][random.randint(0, 2)]]
    images = [Image.open(x) for x in images_list]

    widths, heights = images[0].size

    total_width = widths*2
    max_height = heights*2

    new_im = Image.new('RGB', (total_width, max_height))

    xy = [[0, 0], [0, 1], [1, 0], [1, 1]]
    offset = 0
    for im in images:
        x_offset, y_offset = im.size[0]*xy[offset][0], im.size[0]*xy[offset][1]
        new_im.paste(im, (x_offset, y_offset))
        offset+=1
    
    new_im.save('test.jpg')
    new_im = new_im.resize((40, 40), Image.AFFINE)
    new_im.save('test_lower.jpg') 
    # convert_tensor = transforms.ToTensor()
    # new_im = convert_tensor(new_im)  
    
    new_im = cv2.imread('test_lower.jpg', cv2.IMREAD_GRAYSCALE)

    # g = dgl.graph(([0, 1,  2,  3,  0,  1, 2,  3],
    #                 [1, 0,  3,  2,  2,  3, 0,  1]), num_nodes=4)
    matrix_bo_on = [[None, 0, None], [None, None, None], [2, 2, 2]]
    matrix_on_bo = [[None, None, 4], [1, None, 4], [None, None, 3]]

    # g.ndata['object'] = torch.zeros(4, 3)
    # for i in range(0, 4):
    #     g.ndata['object'][i][x[i][0]] = 1

    # g.edata['relation'] = torch.zeros(8, 7)
    # g.edata['relation'][4][5] = 1
    # g.edata['relation'][5][5] = 1
    # g.edata['relation'][6][6] = 1
    # g.edata['relation'][7][6] = 1


    # g.edata['relation'][0][matrix_bo_on[x[0][0]][x[1][0]]] = 1
    # g.edata['relation'][1][matrix_bo_on[x[2][0]][x[3][0]]] = 1
    # g.edata['relation'][2][matrix_on_bo[x[1][0]][x[0][0]]] = 1
    # g.edata['relation'][3][matrix_on_bo[x[3][0]][x[2][0]]] = 1

  
    edge_index = torch.tensor(([0, 1,  2,  3,  0,  1, 2,  3],
                             [1, 0,  3,  2,  2,  3, 0,  1]), dtype=torch.long).t().contiguous()
    nodes = torch.zeros(4, 3)
    for i in range(0, 4):
        nodes[i][x[i][0]] =1 
    
    edge_atr = torch.zeros(8, 7)
    edge_atr[4][5] = 1
    edge_atr[5][5] = 1
    edge_atr[6][6] = 1
    edge_atr[7][6] = 1


    edge_atr[0][matrix_bo_on[x[0][0]][x[1][0]]] = 1
    edge_atr[1][matrix_bo_on[x[2][0]][x[3][0]]] = 1
    edge_atr[2][matrix_on_bo[x[1][0]][x[0][0]]] = 1
    edge_atr[3][matrix_on_bo[x[3][0]][x[2][0]]] = 1


    data = Data(x=nodes, edge_index=edge_index.t().contiguous(), edge_attr=edge_atr, image=(torch.from_numpy(new_im).float()/255.))





    return data, torch.from_numpy(new_im).float()/255.

data, image = graph_pic_gen()
print(image)