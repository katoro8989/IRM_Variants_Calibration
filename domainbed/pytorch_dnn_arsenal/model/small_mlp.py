import torch.nn as nn
from collections import OrderedDict

'''
Adopted from arxiv.org/pdf/1906.07774.pdf

From Appendix B.2:
> This one is a one hidden layer MLP. Input size is 7 × 7 = 49 and output size is 10. The default number
of hidden units is 70. We use ReLU activations.
'''


def SmallMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden', nn.Linear(49, 70)),
            ('relu_1', nn.ReLU()),
            ('out', nn.Linear(70, num_classes)),
            ('relu_2', nn.ReLU())
        ])
    )


def DeeperSmallMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden_1', nn.Linear(49, 33)),
            ('relu_1', nn.ReLU()),
            ('hidden_2', nn.Linear(33, 33)),
            ('relu_2', nn.ReLU()),
            ('hidden_3', nn.Linear(33, 33)),
            ('relu_3', nn.ReLU()),
            ('out', nn.Linear(33, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )


def EvenDeeperSmallMLP(num_classes=10):
    return nn.Sequential(
        OrderedDict([
            ('flatten', nn.Flatten()),
            ('hidden_1', nn.Linear(49, 20)),
            ('relu_1', nn.ReLU()),
            ('hidden_2', nn.Linear(20, 20)),
            ('relu_2', nn.ReLU()),
            ('hidden_3', nn.Linear(20, 20)),
            ('relu_3', nn.ReLU()),
            ('hidden_4', nn.Linear(20, 20)),
            ('relu_4', nn.ReLU()),
            ('hidden_5', nn.Linear(20, 20)),
            ('relu_5', nn.ReLU()),
            ('hidden_6', nn.Linear(20, 20)),
            ('relu_6', nn.ReLU()),
            ('hidden_7', nn.Linear(20, 20)),
            ('relu_7', nn.ReLU()),
            ('hidden_8', nn.Linear(20, 20)),
            ('relu_8', nn.ReLU()),
            ('out', nn.Linear(20, num_classes)),
            ('out_relu', nn.ReLU())
        ])
    )
