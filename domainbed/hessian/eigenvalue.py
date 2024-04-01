import os
import sys
import torch
import numpy as np
from pyhessian import hessian

sys.path.append(os.environ['PYINFO_PATH'])
from pytorch_infomatrix import fisher_for_cross_entropy_eigenvalues, hessian_eigenvalues
from pytorch_infomatrix import FISHER_MC, COV, HESSIAN
from pytorch_infomatrix import SHAPE_DIAG, SHAPE_FULL
from pytorch_infomatrix import SHAPE_KRON, SHAPE_BLOCK_DIAG

BATCH_SIZE_PER_GPU = 128
PYHESSIAN_SAMPLE_SIZE = 1024



# Calculation of F
def calc_eigenvalue_of_fisher(model, data_loader, top_n):
    """
    Calculate the eigenvalue of FisherMC
    """

    # print("calc fisher")
    stats_name = 'full_batch'
    fisher_type = FISHER_MC
    fisher_shape = SHAPE_FULL
    model.zero_grad()
    eigvals, eigvecs = fisher_for_cross_entropy_eigenvalues(
        model,
        fisher_type,
        fisher_shape,
        data_loader=data_loader,
        top_n=top_n)

    del data_loader
    return eigvals, eigvecs


# Calculation of H
def calc_eigenvalue_of_hessian(model, data_loader, top_n):
    """
    Calculate the eigenvalue of Hessian
    """

    # print("calc hessian")
    stats_name = 'full_batch'
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    model.zero_grad()

    eigvals, eigvecs = hessian_eigenvalues(
        model,
        loss_fn,
        data_loader=data_loader,
        top_n=top_n)

    del data_loader
    return eigvals, eigvecs


# Data Loader for PyHessian
def build_hessian_dataloader(data_loader, hessian_sample_size,
                             hessian_batch_size):
    """
    Build data loader for PyHessian
    """

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
def calc_eigenvalue_of_hessian_from_pyhessian(model, data_loader, top_n):
    """
    Calculate the eigenvalue of Hessian by using PyHessian
    """

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

    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=top_n)
    del hessian_comp
    del data_loader
    del hessian_dataloader
    return top_eigenvalues, top_eigenvectors