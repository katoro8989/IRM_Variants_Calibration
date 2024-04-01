# Reference
# https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py

import attr
from torch.optim import Optimizer
# import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

class _DampingScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_damping', group['damping'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_damping' not in group:
                    raise KeyError("param 'initial_damping' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_dampings = [group['initial_damping'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `damping_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has adampingeady been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_damping(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_damping

    def get_damping(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_damping(self, is_verbose, group, damping, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, damping))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, group, damping))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`damping_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first damping_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `damping_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `damping_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_damping_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_damping_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_damping_called_within_step = False

        with _enable_get_damping_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_damping()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_damping"):
                    values = self._get_closed_form_damping()
                else:
                    values = self.get_damping()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, damping = data
            param_group['damping'] = damping
            self.print_damping(self.verbose, i, damping, epoch)

        self._last_damping = [group['damping'] for group in self.optimizer.param_groups]


class ExponentialDamping(_DampingScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial damping as damping.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialDamping, self).__init__(optimizer, last_epoch, verbose)

    def get_damping(self):
        if not self._get_damping_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_damping()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['damping'] for group in self.optimizer.param_groups]
        return [group['damping'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_damping(self):
        return [base_damping * self.gamma ** self.last_epoch
                for base_damping in self.base_dampings]

class LambdaDamping(_DampingScheduler):
    """Sets the learning rate of each parameter group to the initial damping
    times a given function. When last_epoch=-1, sets initial damping as damping.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        damping_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = Lambdadamping(optimizer, damping_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, damping_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(damping_lambda, list) and not isinstance(damping_lambda, tuple):
            self.damping_lambdas = [damping_lambda] * len(optimizer.param_groups)
        else:
            if len(damping_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} damping_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(damping_lambda)))
            self.damping_lambdas = list(damping_lambda)
        super(LambdaDamping, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'damping_lambdas')}
        state_dict['damping_lambdas'] = [None] * len(self.damping_lambdas)

        for idx, fn in enumerate(self.damping_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['damping_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        damping_lambdas = state_dict.pop('damping_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['damping_lambdas'] = damping_lambdas

        for idx, fn in enumerate(damping_lambdas):
            if fn is not None:
                self.damping_lambdas[idx].__dict__.update(fn)

    def get_damping(self):
        if not self._get_damping_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_damping()`.")

        return [base_damping * lmbda(self.last_epoch)
                for lmbda, base_damping in zip(self.damping_lambdas, self.base_dampings)]

class StepDamping(_DampingScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial damping as damping.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses damping = 0.05 for all groups
        >>> # damping = 0.05     if epoch < 30
        >>> # damping = 0.005    if 30 <= epoch < 60
        >>> # damping = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = Stepdamping(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepDamping, self).__init__(optimizer, last_epoch, verbose)

    def get_damping(self):
        if not self._get_damping_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_damping()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['damping'] for group in self.optimizer.param_groups]
        return [group['damping'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_damping(self):
        return [base_damping * self.gamma ** (self.last_epoch // self.step_size)
                for base_damping in self.base_dampings]

@attr.s
class DampingSchedulerSetting:
    name = attr.ib()
    optimizer = attr.ib()
    gamma = attr.ib()
    step_size = attr.ib()
    max_iteration = attr.ib()
    # h_params = attr.ib()


def build_damping_scheduler(setting: DampingSchedulerSetting):

    name = setting.name
    optimizer = setting.optimizer
    gamma = setting.gamma
    step_size = setting.step_size

    if name == "exponential":
        scheduler = ExponentialDamping(optimizer=optimizer, gamma=gamma)
        return scheduler
    elif name == "invsqrt":
        scheduler = LambdaDamping(
            optimizer=optimizer, damping_lambda=lambda steps: 1/math.sqrt(steps+1))
        return scheduler
    elif name == "step":
        scheduler = StepDamping(
            optimizer=optimizer, step_size=step_size, gamma=gamma)
        return scheduler
    elif name == "constant":
        scheduler = LambdaDamping(
            optimizer=optimizer, damping_lambda=lambda x: 1.0)
        return scheduler
    else:
        raise ValueError(f'{name} is not supported')

