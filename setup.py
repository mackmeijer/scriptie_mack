import argparse
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F
import egg.core as core
import torch.nn as nn
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from dataloaders import create_data_loaders
from typing import Any, List, Optional, Sequence, Union
from torchvision import models
from PIL import Image
from torchvision.utils import save_image
import torch.utils.data
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset

torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")
class Collater:
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.game_size = game_size
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch = batch[:((len(batch) // self.game_size) * self.game_size)]  # we throw away the last batch_size % game_size
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # we return a tuple (sender_input, labels, receiver_input, aux_input)
            # we use aux_input to store minibatch of graphs
            return (
                torch.zeros(len(batch) // self.game_size, 1),  # we don't need sender_input --> create a fake one
                torch.zeros(len(batch) // self.game_size).long(),  # the target is aways the first graph among game_size graphs
                None,  # we don't care about receiver_input
                batch  # this is a compact data for batch_size graphs 
            )

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.game_size = game_size
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(game_size, dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size):
        super().__init__()
        self.head_size = 2
        self.game_size = game_size
        feature_size = 1600

        self.dropout = nn.Dropout(0.2)
        self.resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.fc = nn.Linear(512, 100, bias=True)

        self.gr_conv1 = GATv2Conv(num_node_features, n_hidden, edge_dim = 7,  num_heads = self.head_size)
        self.gr_conv2 = GATv2Conv(n_hidden, n_hidden, edge_dim = 7, num_heads = self.head_size)
        self.emb_lin1 = nn.Linear(feature_size, 50, bias=False)

        self.lin1 = nn.Linear(1600, 400)
        self.lin2 = nn.Linear(400, 100)
        self.lin3 = nn.Linear(100, n_hidden)


    def forward(self, data):
        x, edge_index, edge_feat, image = data.x, data.edge_index, data.edge_attr, data.image  # check number of graphs via: data.num_graphs
        
        x = self.gr_conv1(x, edge_index, edge_feat)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = scatter(x, data.batch, dim=0, reduce='mean')  # size: data.num_graphs * n_hidden

        emb = self.return_embeddings(image, data.num_graphs, self.game_size) # num_graphs, 1, 1, embedding_size

        a = emb.squeeze(dim = 1).flatten(1, 2)
        a = self.lin1(a)
        a = F.leaky_relu(a)
        a = self.dropout(a)
        a = self.lin2(a)
        a = F.leaky_relu(a)
        a = self.dropout(a)
        a = self.lin3(a)
        a = F.tanh(a)
        a = self.dropout(a)

        h = self.im_lin3(torch.cat((a, x), dim = 1))  # size: data.num_graphs * n_hidden

        return a

    def return_embeddings(self, x, num_graphs, game_size):
        # Embed each image (left or right)
        embs = []


        h = x.chunk(num_graphs)
        for i in range(num_graphs):
            b = h[i]
            # b = self.im_conv0(b.reshape(40, 40, 1))   
            # b = self.resnet18(b.unsqueeze(dim = 0))
            embs.append(b.detach())

        x = torch.stack(embs).flatten().view(num_graphs, 1,  40, 40)

        return x
    
class GCN2(torch.nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size):
        super().__init__()
        self.head_size = 2
        self.game_size = game_size
        feature_size = 1600

        self.gr_conv1 = GATv2Conv(num_node_features, n_hidden, edge_dim = 7,  num_heads = self.head_size)
        self.gr_conv2 = GATv2Conv(n_hidden, n_hidden, edge_dim = 7, num_heads = self.head_size)
        self.emb_lin1 = nn.Linear(feature_size, 50, bias=False)

        self.im_conv1 = nn.Conv2d(1, n_hidden, kernel_size=1, stride=1, bias=False)
        self.im_conv2 = nn.Conv2d(1, 1, kernel_size=(n_hidden, 1), stride=(n_hidden, 1), bias=False)
        self.im_lin1 = nn.Linear(50, n_hidden, bias=False)
        self.im_lin2 = nn.Linear(n_hidden, n_hidden, bias=False)

        self.im_lin3 = nn.Linear(n_hidden*2, n_hidden, bias=False)

    def forward(self, data):
        x, edge_index, edge_feat, image = data.x, data.edge_index, data.edge_attr, data.image  # check number of graphs via: data.num_graphs
        # emb = self.return_embeddings(image, data.num_graphs, self.game_size)
        x = self.gr_conv1(x, edge_index, edge_feat)
        x = self.gr_conv2(x, edge_index, edge_feat)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = scatter(x, data.batch, dim=0, reduce='mean')  # size: data.num_graphs * n_hidden
        # h = self.im_conv1(emb)
        # h = torch.nn.LeakyReLU()(h)
        # h = h.transpose(1, 2)   
        # h = self.im_conv2(h)                       # batch_size, 1, 1, embedding_size
        # h = torch.nn.LeakyReLU()(h)
        # h = h.squeeze()                         # batch_size x embedding_size
        # h = self.im_lin1(h)                        # batch_size x hidden_size
        # h = h.mul(1.0 / 0.01)
        # h = self.im_lin3(torch.cat((h, x), dim = 1))
        
        return x

    def return_embeddings(self, x, num_graphs, game_size):
        # Embed each image (left or right)
        embs = []


        h = x.chunk(num_graphs)
        for i in range(num_graphs):
            b = h[i].view(-1)
            b = self.emb_lin1(b)

            embs.append(b.detach())
        x = torch.stack(embs).flatten().view(num_graphs, 1,  1, 30)
  
        return x
    
class Sender(nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size):
        super().__init__()
        self.game_size = game_size
        self.gcn = GCN(num_node_features, n_hidden, game_size)
        self.fc1 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, _aux_input):
        # _aux_input is a minibatch of n_games x game_size graphs
        # we don't care about x
        data = _aux_input
        assert data.num_graphs % self.game_size == 0
        x = self.gcn(data)[::self.game_size]  # we just need the target graph, hence we take only the first graph of every game_size graphs
        return self.fc1(x)  # size: n_games * n_hidden  (note: n_games = batch_size // game_size)


class Receiver(nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size):
        super().__init__()
        self.game_size = game_size

        self.gcn = GCN(num_node_features, n_hidden, game_size)
        self.fc1 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, _input, _aux_input):
        # x is the tensor of shape n_games * n_hidden -- each row is an embedding decoded from the message sent by the sender
        cands = self.gcn(_aux_input)  # graph embeddings for all batch_size graphs; size: batch_size * n_hidden
        cands = cands.view(cands.shape[0] // self.game_size, self.game_size, -1)  # size: n_games * game_size * n_hidden 
        dots = torch.matmul(cands, torch.unsqueeze(x, dim=-1))  # size: n_games * game_size * 1
        return dots.squeeze()  # size: n_games * game_size: each row is a list of scores for a game (each score tells how good the corresponding candidate is) 


def get_params(params):
    parser = argparse.ArgumentParser()

    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=20,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=20,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=5,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=5,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    parser.add_argument(
        "--game_size",
        type=int,
        default=2,
        help="The number of graphs in a game (including a target and distractors) (default: 4)",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    # opts.batch_size = 4
    print(opts, flush=True)
    game_size = opts.game_size

    # we care about the communication success: the accuracy that the receiver can distinguish the target from distractors
    def loss(
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

    # we create dataset and dataloader

    doubleset1,  val_loader1, dataset1 = create_data_loaders(1)
    doubleset2,  val_loader1, dataset2 = create_data_loaders(1)
    train_loader = DataLoader(game_size, dataset1, batch_size=opts.batch_size, shuffle=True)
    val_loader = DataLoader(game_size, dataset2, batch_size=opts.batch_size, shuffle=True)



    # we create the two agents
    receiver = Receiver(3, n_hidden=opts.receiver_hidden, game_size=game_size) 
    sender = Sender(3, n_hidden=opts.sender_hidden, game_size=game_size)

    sender = core.RnnSenderReinforce(
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell,
    )
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        receiver_entropy_coeff=0,
    )
    callbacks = []

    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks
        + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
    )

    trainer.train(n_epochs=opts.n_epochs)

main(["--n_epochs", "250", "--max_len", "1", "--lr", "1e-3", "--batch_size", "8", "--vocab_size", "100"])
