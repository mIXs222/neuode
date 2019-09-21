"""
Function dump
"""

import torch.nn as nn

from neuode.interface.struct import ActivationFn


# dictionary from enum to nn
ACTFN_NN_DICT = {
    ActivationFn.NONE: nn.Identity,
    ActivationFn.THRESHOLD: nn.Threshold,
    ActivationFn.RELU: nn.ReLU,
    ActivationFn.RRELU: nn.RReLU,
    ActivationFn.HARDTANH: nn.Hardtanh,
    ActivationFn.RELU6: nn.ReLU6,
    ActivationFn.SIGMOID: nn.Sigmoid,
    ActivationFn.TANH: nn.Tanh,
    ActivationFn.ELU: nn.ELU,
    ActivationFn.SELU: nn.SELU,
    ActivationFn.GLU: nn.GLU,
    ActivationFn.HARDSHRINK: nn.Hardshrink,
    ActivationFn.LEAKYRELU: nn.LeakyReLU,
    ActivationFn.LOGSIGMOID: nn.LogSigmoid,
    ActivationFn.SOFTPLUS: nn.Softplus,
    ActivationFn.SOFTSHRINK: nn.Softshrink,
    ActivationFn.PRELU: nn.PReLU,
    ActivationFn.SOFTSIGN: nn.Softsign,
    ActivationFn.TANHSHRINK: nn.Tanhshrink,
    ActivationFn.SOFTMIN: nn.Softmin,
    ActivationFn.SOFTMAX: nn.Softmax,
    ActivationFn.SOFTMAX2D: nn.Softmax2d,
    ActivationFn.LOGSOFTMAX: nn.LogSoftmax,
}


# translastte from activation fn enum to torch nn
def actfn2nn(act_fn):
    return ACTFN_NN_DICT[act_fn]()


# wrap time into vector x
def wrap_time_vec(t, x):
    t_aug = torch.ones(x.shape[0], 1) * t
    return torch.cat([x, aug], 1)

# wrap time into matrix x
def wrap_time_img(t, x):
    batch_size, channels, height, width = x.shape
    t_aug = torch.ones(batch_size, 1, height, width) * t
    return torch.cat([x, aug], 1)