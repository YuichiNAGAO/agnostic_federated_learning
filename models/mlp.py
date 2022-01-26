
import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ["mlp"]




class MLP(nn.Module):
    def __init__(self, in_features=32*32*3, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def mlp(args):
    dataset = args.dataset

    if "cifar" in dataset or  dataset == "mnist" or dataset == "fmnist":
        return MLP()
    else:
        raise NotImplementedError(f"not supported yet.")