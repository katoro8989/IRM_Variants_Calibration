# coding; utf-8
from functools import partial
import attr
from torch import Generator
from torch.utils.data import random_split
from torchvision import datasets, transforms


@attr.s
class DatasetSetting:
    name = attr.ib()
    root = attr.ib()
    split_ratio = attr.ib(default=0.8)
    split_seed = attr.ib(default=1)
    # For ImageNet
    # root_val = attr.ib(default=None)


@attr.s
class Dataset:
    name = attr.ib()
    num_classes = attr.ib()
    train_dataset = attr.ib()
    val_dataset = attr.ib()
    test_dataset = attr.ib()


def _transformer(data_name, train=True):
    if data_name == 'tiny_mnist':
        return transforms.Compose([
            transforms.Resize((7, 7)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if data_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if data_name == 'fashion_mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    if data_name == 'svhn':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
    if data_name == 'cifar10':
        if train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    if data_name == 'cifar100':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    if data_name == 'imagenet':
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


def _split_train_val(train_val_dataset, split_ratio:float, split_seed:int):
    n_samples = len(train_val_dataset)
    train_size = int(n_samples * split_ratio)
    val_size = n_samples - train_size
    return random_split(
        train_val_dataset, [train_size, val_size], generator=Generator().manual_seed(split_seed)
    )


def build_dataset(setting: DatasetSetting) -> Dataset:
    name = setting.name
    root = setting.root
    split_ratio = setting.split_ratio
    split_seed = setting.split_seed
    datasets_f = {
        'tiny_mnist': partial(datasets.MNIST, download=True),
        'mnist': partial(datasets.MNIST, download=True),
        'fashion_mnist': partial(datasets.FashionMNIST, download=True),
        'svhn': partial(datasets.SVHN, download=True),
        'cifar10': partial(datasets.CIFAR10, download=True),
        'cifar100': partial(datasets.CIFAR100, download=True),
        'imagenet': None
    }[name]

    if name == 'imagenet':        
        train_root = "{}/train".format(root)
        test_root = "{}/val".format(root)

        train_val_dataset = \
            datasets.ImageFolder(root=train_root,
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)


        test_dataset = \
            datasets.ImageFolder(root=test_root,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))

        print(f'len(train_dataset): {len(train_dataset)}')
        print(f'len(val_dataset): {len(val_dataset)}')
        print(f'len(test_dataset): {len(test_dataset)}')

        return Dataset(name, 1000, train_dataset, val_dataset, test_dataset)
    
    if name == 'svhn':
        train_val_dataset = datasets_f(
            root, split='train', transform=_transformer(name, True))
        train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)
        test_dataset = datasets_f(
            root, split='test', transform=_transformer(name, False))

        print(f'len(train_dataset): {len(train_dataset)}')
        print(f'len(val_dataset): {len(val_dataset)}')
        print(f'len(test_dataset): {len(test_dataset)}')
        return Dataset(name, 10, train_dataset, val_dataset, test_dataset)

    train_val_dataset = datasets_f(
        root, train=True, transform=_transformer(name, True))
    train_dataset, val_dataset = _split_train_val(train_val_dataset, split_ratio, split_seed)
    test_dataset = datasets_f(
        root, train=False, transform=_transformer(name, False))

    print(f'len(train_dataset): {len(train_dataset)}')
    print(f'len(val_dataset): {len(val_dataset)}')
    print(f'len(test_dataset): {len(test_dataset)}')
    return Dataset(name, 100 if name == 'cifar100' else 10, train_dataset, val_dataset, test_dataset)
