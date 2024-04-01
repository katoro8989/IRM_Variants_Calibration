# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
from collections import Counter

# Added by Hiroki Naganuma
from domainbed.calibration.ece import calc_ece
from domainbed.calibration.utils import get_maxprob_and_onehot

# Added by Kotaro Yoshida
from torch import nn

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def evaluate(network, loader, ece_config, device):
    correct = 0
    total = 0

    labels_list = []
    probs_list = []

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            input = x.to(device)
            labels = y.to(device)
            logits = network.predict(input)
            pred_label = logits.argmax(dim=1)

            # Calculate Accuracy
            total += labels.size(0)
            correct += (pred_label == labels).sum().item()

            # Calculate ECE
            probs = torch.nn.functional.softmax(logits, dim=1)
            labels_list.extend(labels.detach().cpu().numpy())
            probs_list.extend(probs.detach().cpu().numpy())

    all_probs = np.array(probs_list)
    all_labels = np.array(labels_list)
    maxprob_list, one_hot_labels = get_maxprob_and_onehot(all_probs, all_labels)
    ece_config['num_samples'] = int(len(one_hot_labels))
    avg_ece = calc_ece(ece_config, maxprob_list, one_hot_labels)
    
    network.train()
    del probs_list, labels_list, all_probs, all_labels, maxprob_list, one_hot_labels

    avg_acc = 100 * correct / total
    return avg_acc, avg_ece

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                # Won't Use
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)

            batch_weights = batch_weights.to(device)
            if logits.size(1) == 1:
                # Won't Use
                correct += (logits.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
                pred_label = logits.argmax(dim=1)
                correct_preds = pred_label.eq(y)
                weighted_correct_preds = correct_preds.float() * batch_weights
                correct += weighted_correct_preds.sum().item()

            total += batch_weights.sum().item()
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
    
# Embedding layer for BIRM
class EBD(nn.Module):
    def __init__(self, num_domains, num_classes):
      super(EBD, self).__init__()
      self.num_domains = num_domains
      self.num_classes = num_classes
      self.embedings = torch.nn.Embedding(self.num_domains, self.num_classes)
      self.re_init()

    def re_init(self):
      self.embedings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd):
      rd = torch.normal(
         torch.Tensor([1.0] * self.num_domains * self.num_classes),
         torch.Tensor([noise_sd] * self.num_domains * self.num_classes))
      self.embedings.weight.data = rd.view(-1, self.num_classes).cuda()

    def forward(self, e):
      return self.embedings(e.long())