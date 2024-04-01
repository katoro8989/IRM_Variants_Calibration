""" Layer-wise adaptive rate scaling for SGD in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required
# import track


class SGD_Analyze(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS_Clipping coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = SGD_Analyze(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_epoch=200, betas=(0.998, 0.99)):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError(
                "Invalid LARS_Clipping coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch, betas=betas)
        super(SGD_Analyze, self).__init__(params, defaults)

    def step(self, iteration=None, epoch=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']

            # layer番号
            idx_layer = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0

                param_state['step'] += 1

                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                decay = 1
                global_lr = lr * decay

                # local_lr = eta * weight_norm / \
                #     (grad_norm + weight_decay * weight_norm)

                # To pretend SGD
                local_lr = 1

                # Perform as LARS
                actual_lr = float(local_lr * global_lr)

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                        torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)

                # moment_t = actual_lr * (d_p + weight_decay * p.data)
                # moment_t_norm = float(torch.norm(moment_t).cpu())
                # weight_norm=float(weight_norm.cpu())
                # grad_norm=float(grad_norm.cpu())
                # # local_lr=float(local_lr.cpu())

                # track.metric(iteration=iteration, epoch=epoch,
                #              idx_layer=idx_layer,
                #              weight_norm=weight_norm,
                #              grad_norm=grad_norm,
                #              local_lr=local_lr,
                #              moment_norm=moment_t_norm,
                #              global_lr=global_lr,
                #              actual_lr=actual_lr)
                idx_layer += 1

        return loss
