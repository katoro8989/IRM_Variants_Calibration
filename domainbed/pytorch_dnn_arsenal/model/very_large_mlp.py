import torch.nn as nn
from collections import OrderedDict


# Number of params = 33610
def VeryLargeMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden', nn.Linear(49, 560)),
            ('relu_1', nn.ReLU()),
            ('out', nn.Linear(560, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )

