from collections import OrderedDict
import torch.nn as nn

def SimpleCNN(num_classes=10, dropout_ratio=0.5, width_factor=1, dataset='MNIST'):
    in_dim = 3 if dataset == 'SVHN' else 1
    if dataset == 'SVHN':
        fc_in_dim = 64 * width_factor * 16 * 16
    else:
        fc_in_dim = 64 * width_factor * 14 * 14

    modules = OrderedDict(
        [
            ('conv1', nn.Conv2d(in_dim, 32 * width_factor, kernel_size=5, stride=1, padding=2)),
            ('relu1', nn.ReLU()),
            (
                'conv2',
                nn.Conv2d(32 * width_factor, 64 * width_factor, kernel_size=5, stride=1, padding=2)
            ),
            ('relu2', nn.ReLU()),
            ('max_pool1', nn.MaxPool2d(2, stride=2)),
            ('flatten', nn.Flatten()),
            ('dropout1', nn.Dropout(dropout_ratio)),
            ('linear1', nn.Linear(fc_in_dim, 1024 * width_factor)),
            ('relu3', nn.ReLU()),
            ('dropout2', nn.Dropout(dropout_ratio)),
            ('linear2', nn.Linear(1024 * width_factor, num_classes)),
        ]
    )

    model = nn.Sequential(modules)

    return model
