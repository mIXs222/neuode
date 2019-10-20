"""
Convolution-based dynamics
"""

import torch.nn as nn

from neuode.interface.common import DynamicMap
from neuode.interface.struct import ConvSpec
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