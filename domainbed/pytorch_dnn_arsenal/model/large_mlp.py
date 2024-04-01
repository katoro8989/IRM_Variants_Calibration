import torch.nn as nn
from collections import OrderedDict


# Number of params = 16810
def LargeMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden', nn.Linear(49, 280)),
            ('relu_1', nn.ReLU()),
            ('out', nn.Linear(280, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )

