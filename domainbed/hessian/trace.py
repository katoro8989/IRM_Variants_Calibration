import os
import sys
import torch
import numpy as np
from pyhessian import hessian

sys.path.append(os.environ['PYINFO_PATH'])
from pytorch_infomatrix import fisher_for_cross_entropy, hessian_for_loss
from pytorch_infomatrix import FISHER_MC, COV, HESSIAN
from pytorch_infomatrix import SHAPE_DIAG, SHAPE_FULL
from pytorch_infomatrix import SHAPE_KRON, SHAPE_BLOCK_DIAG

BATCH_SIZE_PER_GPU = 128
PYHESSIAN_SAMPLE_SIZE = 1024


# Calculation of F
def calc_trace_of_fisher(model, data_loader):
    # print("calc fisher")
    stats_name = 'full_batch'
    fisher_type = FISHER_MC
    fisher_shape = SHAPE_DIAG
    model.zero_grad()
    
    matrix_manager = fisher_for_cross_entropy(model,
                                              fisher_type,
                                              fisher_shape,
                                              data_loader=data_loader,
                                              stats_name=stats_name)

    trace_f = matrix_manager.get_trace(fisher_type, fisher_shape, stats_name)
    matrix_manager.clear_matrices(stats_name)

    del matrix_manager
    del data_loader
    return trace_f


# Calculation of H
def calc_trace_of_hessian(model, data_loader):
    # print("calc hessian")
    stats_name = 'full_batch'
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    hessian_type = HESSIAN
    hessian_shape = SHAPE_DIAG
    model.zero_grad()

    matrix_manager = hessian_for_loss(model,
                                      loss_fn,
                                      hessian_shape,
                                      data_loader=data_loader,
                                      stats_name=stats_name,
                                      matrix_manager=None)

    trace_h = matrix_manager.get_trace(hessian_type, hessian_shape, stats_name)
    matrix_manager.clear_matrices(stats_name)

    del matrix_manager
    del data_loader
    return trace_h


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
def calc_trace_of_hessian_from_pyhessian(model, data_loader):
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
    del hessian_comp
    del data_loader
    del hessian_dataloader
    return trace_h