"""
Convolution-based dynamics
"""

import numpy as np
import torch
import torch.nn as nn

from neuode.interface.common import *
from neuode.interface.struct import *
import neuode.util as util


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


class FullConvDMap(DynamicMap):

    def __init__(self, specs):
        super(DynamicMap, self).__init__()

        # sanity check
        assert len(specs) > 0

        # construct each linear layer
        self.nets = nn.ModuleList([ConvDMap(spec) for spec in specs])


    def forward(self, t, x, *args, **kwargs):
        for net in self.nets:
            x = net(t, x, *args, **kwargs)
        return x