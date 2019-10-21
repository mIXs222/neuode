"""
Convolution-based dynamics
"""

import torch
import torch.nn as nn

from neuode.interface.common import DynamicMap
from neuode.interface.struct import ConvSpec, ActivationFn
import neuode.util.util as util


# Translational filter
class ConvDMap(DynamicMap):

    def __init__(self, spec):
        super(DynamicMap, self).__init__()

        # sanity check
        assert isinstance(spec, ConvSpec)

        # time dependent
        self.use_time = spec.use_time
        in_channel_add = 1 if self.use_time else 0

        # parse spec into linear layer
        nets = []
        nets.append(nn.Conv2d(
            spec.in_channel + in_channel_add,
            spec.out_channel,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
        ))
        nets.append(util.actfn2nn(spec.act_fn))
        self.net = nn.Sequential(*nets)

    def forward(self, t, x, *args, **kwargs):
        if self.use_time:
            x = util.wrap_time_img(t, x)
        return self.net(x)


# from rtqichen/ffjord/blob/master/lib/layers/diffeq_layers/basic.py
class ConcatSquashConv2d(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            ksize=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            transpose=False,
            actfn=ActivationFn.NONE):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._actfn = util.actfn2nn(actfn)

    def forward(self, t, x):
        x = self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(
            1, -1, 1, 1) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)
        return self._actfn(x)