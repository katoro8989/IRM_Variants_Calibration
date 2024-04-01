# coding: utf-8
import attr
import torch


@attr.s
class CriterionSetting:
    name = attr.ib(default='')


def build_criterion(_: CriterionSetting = CriterionSetting()):
    return torch.nn.CrossEntropyLoss()
