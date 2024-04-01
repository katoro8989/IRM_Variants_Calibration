import torch.nn as nn
from collections import OrderedDict


# Number of params = 8410
def MediumMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden', nn.Linear(49, 140)),
            ('relu_1', nn.ReLU()),
            ('out', nn.Linear(140, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )


def DeeperMediumMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden_1', nn.Linear(49, 50)),
            ('relu_1', nn.ReLU()),
            ('hidden_2', nn.Linear(50, 50)),
            ('relu_2', nn.ReLU()),
            ('hidden_3', nn.Linear(50, 50)),
            ('relu_3', nn.ReLU()),
            ('out', nn.Linear(50, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )


def EvenDeeperMediumMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden_1', nn.Linear(49, 30)),
            ('relu_1', nn.ReLU()),
            ('hidden_2', nn.Linear(30, 30)),
            ('relu_2', nn.ReLU()),
            ('hidden_3', nn.Linear(30, 30)),
            ('relu_3', nn.ReLU()),
            ('hidden_4', nn.Linear(30, 30)),
            ('relu_4', nn.ReLU()),
            ('hidden_5', nn.Linear(30, 30)),
            ('relu_5', nn.ReLU()),
            ('hidden_6', nn.Linear(30, 30)),
            ('relu_6', nn.ReLU()),
            ('hidden_7', nn.Linear(30, 30)),
            ('relu_7', nn.ReLU()),
            ('hidden_8', nn.Linear(30, 30)),
            ('relu_8', nn.ReLU()),
            ('out', nn.Linear(30, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )
