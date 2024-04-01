# coding: utf-8
import attr
import torch.nn
from torchvision import models

from .all_cnn_c import ALL_CNN_C
from .resnet8 import ResNet8
from .vgg import VGG11
from .simple_cnn import SimpleCNN
from .small_mlp import SmallMLP, DeeperSmallMLP, EvenDeeperSmallMLP
from .medium_mlp import MediumMLP, DeeperMediumMLP, EvenDeeperMediumMLP
from .large_mlp import LargeMLP
from .very_large_mlp import VeryLargeMLP

from .nn import NN, NN_SC, NN_tiny3, NN_SC_tiny3
from .lnn import LNN, LNN_SC, LNN_tiny3, LNN_SC_tiny3, LNN_CIFAR, LNN_SC_CIFAR

@attr.s
class ModelSetting:
    name = attr.ib()
    num_classes = attr.ib()
    dropout_ratio = attr.ib()


def build_model(setting: ModelSetting):
    # ResNet8, SimpleCNN
    name = setting.name
    num_classes = setting.num_classes
    dropout_ratio = setting.dropout_ratio

    if name == 'all_cnn_c':
        model = ALL_CNN_C(num_classes)
        return model
    if name == 'vgg16':
        model = models.vgg16(pretrained=False, num_classes=num_classes)
        return model
    if name == 'vgg11':
        model = VGG11()
        return model
    if name == 'resnet8':
        model = ResNet8(num_classes)
        return model
    if name == 'resnet18':
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, num_classes)
        return model
    if name == 'resnet50':
        model = models.resnet50()
        model.fc = torch.nn.Linear(2048, num_classes)
        return model
    if name == 'simple_cnn':
        model = SimpleCNN(num_classes, dropout_ratio)
        return model
    if name == 'small_mlp':
        model = SmallMLP(num_classes)
        return model
    if name == 'deeper_small_mlp':
        model = DeeperSmallMLP(num_classes)
        return model
    if name == 'even_deeper_small_mlp':
        model = EvenDeeperSmallMLP(num_classes)
        return model
    if name == 'medium_mlp':
        model = MediumMLP(num_classes)
        return model
    if name == 'deeper_medium_mlp':
        model = DeeperMediumMLP(num_classes)
        return model
    if name == 'even_deeper_medium_mlp':
        model = EvenDeeperMediumMLP(num_classes)
        return model
    if name == 'large_mlp':
        model = LargeMLP(num_classes)
        return model
    if name == 'very_large_mlp':
        model = VeryLargeMLP(num_classes)
        return model
    if name == 'lnn':
        model = LNN()
        return model
    if name == 'lnn_tiny3':
        model = LNN_tiny3()
        return model
    if name == 'lnn_sc':
        model = LNN_SC()
        return model
    if name == 'lnn_sc_tiny3':
        model = LNN_SC_tiny3()
        return model
    if name == 'lnn_cifar':
        model = LNN_CIFAR()
        return model
    if name == 'lnn_sc_cifar':
        model = LNN_SC_CIFAR()
        return model
    if name == 'nn':
        model = NN()
        return model
    if name == 'nn_tiny3':
        model = NN_tiny3()
        return model
    if name == 'nn_sc':
        model = NN_SC()
        return model
    if name == 'nn_sc_tiny3':
        model = NN_SC_tiny3()
        return model
    raise ValueError('The selected model is not supported for this trainer.')
