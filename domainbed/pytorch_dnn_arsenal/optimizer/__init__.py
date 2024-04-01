# coding: utf-8
import attr
import torch.optim as optim

# from .lars import LARS  # noqa: F401
# from .adabound import AdaBound
# from .kfac import KFACOptimizer

from .conjugate_gradient import ConjugateGradientOptimizer
from .sam import SAM
import torch_optimizer

@attr.s
class OptimizerSetting:
    name = attr.ib()
    lr = attr.ib()
    weight_decay = attr.ib()
    model = attr.ib()

    # Momentum
    momentum = attr.ib(default=0.9) # sgd, sgd_nesterov

    # RMSprop
    alpha = attr.ib(default=0.99) # rmsprop (smoothing constant)
    eps = attr.ib(default=0.001) # adam, rmsprop (term added to the denominator to improve numerical stability )
   
    # Adam
    beta_1 = attr.ib(default=0.9) #adam
    beta_2 = attr.ib(default=0.999) #adam

    # LARS
    eta = attr.ib(default=0.001) # lars coefficient

    # KFAC
    damping = attr.ib(default=0.001) 
    
    # CGD
    beta_update_rule = attr.ib(default='FR')
    beta_momentum_coeff = attr.ib(default=1)
    mu = attr.ib(2)
    max_epoch = attr.ib(200)

    # SAM
    epsilon = attr.ib(default=1e-4) #eps
    rho = attr.ib(default=0.05) #rho

    # Shampoo
    update_freq=attr.ib(default=1) 


def build_optimizer(setting: OptimizerSetting):
    name = setting.name
    model_params = setting.model.parameters()
    model = setting.model

    # Standard Optimizer
    if name == 'vanilla_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        weight_decay=setting.weight_decay)

    elif name == 'momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum,
                        weight_decay=setting.weight_decay)

    elif name == 'nesterov_momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum, 
                        weight_decay=setting.weight_decay, 
                        nesterov=True)

    elif name == 'adam':
        return optim.Adam(params=model_params, 
                        lr=setting.lr, 
                        betas=(setting.beta_1, setting.beta_2), 
                        eps=setting.eps, 
                        weight_decay=setting.weight_decay, 
                        amsgrad=False)

    elif name == 'rmsprop':
        return optim.RMSprop(params=model_params, 
                            lr=setting.lr, 
                            alpha=setting.alpha, 
                            eps=setting.eps, 
                            weight_decay=setting.weight_decay, 
                            momentum=setting.momentum)

    elif name == 'cgd':
        return ConjugateGradientOptimizer(params=model_params, 
                                          lr=setting.lr, 
                                          weight_decay=setting.weight_decay, 
                                          beta_update_rule=setting.beta_update_rule, 
                                          beta_momentum_coeff=setting.beta_momentum_coeff,
                                          mu=setting.mu)

    elif name == 'sam':
        return SAM(params=model_params, 
                   base_optimizer=optim.SGD,
                   rho=setting.rho,
                   eps=setting.eps,
                   lr=setting.lr, 
                   momentum=setting.momentum)

    elif name == 'adamw':
        return optim.AdamW(params=model_params, 
                           lr=setting.lr, 
                           betas=(setting.beta_1, setting.beta_2), 
                           eps=setting.eps, 
                           weight_decay=setting.weight_decay, 
                           amsgrad=False)

    elif name == 'shampoo':
        return torch_optimizer.Shampoo(model_params,
                                       lr=setting.lr, 
                                       momentum=setting.momentum,
                                       weight_decay=setting.weight_decay, 
                                       epsilon=setting.epsilon,
                                       update_freq=setting.update_freq)

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')
