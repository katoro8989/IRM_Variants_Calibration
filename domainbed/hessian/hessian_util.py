import os
import sys
import torch
import numpy as np
from pyhessian import hessian

BATCH_SIZE_PER_GPU = 128
PYHESSIAN_SAMPLE_SIZE = 1024


# Data Loader for PyHessian
def build_hessian_dataloader(data_loader, hessian_sample_size,
                             hessian_batch_size):
    batch_num = hessian_sample_size // hessian_batch_size

    if batch_num == 1:
        for inputs, labels in data_loader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(data_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break

    return hessian_dataloader


# Calculation of H from PyHessian
def calc_hessian_from_pyhessian(model, data_loader, top_n):
    # print("calc hessian by using pyhessian")
    model.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    # computing hessian spectrum
    hessian_sample_size = PYHESSIAN_SAMPLE_SIZE  # (num of data for using)
    hessian_dataloader = build_hessian_dataloader(data_loader,
                                                  hessian_sample_size,
                                                  BATCH_SIZE_PER_GPU)
    hessian_comp = hessian(model,
                           loss_fn,
                           dataloader=hessian_dataloader,
                           cuda=True)

    traces = hessian_comp.trace()
    trace_h = np.mean(traces)

    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_n)

    del hessian_comp
    del data_loader
    del hessian_dataloader

    return trace_h, top_eigenvalues

