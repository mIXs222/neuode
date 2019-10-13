"""
Linear dynamic mappings
"""

import torch.nn as nn

from neuode.interface.common import DynamicMap
from neuode.interface.struct import LinearSpec
import neuode.util.util as util


# Fully feed forward layer
class LinearDMap(DynamicMap):

    def __init__(self, spec):
        super(DynamicMap, self).__init__()

        # sanity check
        assert isinstance(spec, LinearSpec)

        # time dependent
        self.use_time = spec.use_time
        in_dim_add = 1 if self.use_time else 0

        # parse spec into linear layer
        nets = []
        nets.append(nn.Linear(spec.in_dim + in_dim_add, spec.out_dim))
        nets.append(util.actfn2nn(spec.act_fn))
        self.net = nn.Sequential(*nets)


    def forward(self, t, x, *args, **kwargs):
        if self.use_time:
            x = util.wrap_time_vec(t, x)
        return self.net(x)