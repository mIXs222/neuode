"""
Linear dynamic mappings
"""

import numpy as np

from neuode.interface.common import *
from neuode.interface.struct import *
import neuode.util as util

class LinearDMap(DynamicMap):

    def __init__(self, spec):
        super(DynamicMap, self).__init__()

        # sanity check
        assert isinstance(spec, LinearSpec)

        # parse spec into linear layer
        nets = []
        nets.append(nn.Linear(spec.in_dim, spec.out_dim))
        nets.append(util.actfn2nn(spec.act_fn))
        self.net = nn.Sequential(*nets)

        # time dependent
        self.use_time = spec.use_time


    def forward(self, t, x, *args, **kwargs):
        if self.use_time:
            x = util.wrap_time_vec(t, x)
        return self.net(x)


class MLPDMap(DynamicMap):

    def __init__(self, specs):
        super(DynamicMap, self).__init__()

        # sanity check
        assert len(specs) > 0

        # construct each linear layer
        self.nets = nn.ModuleList([LinearDMap(spec) for spec in specs])


    def forward(self, t, x, *args, **kwargs):
        for net in self.nets:
            x = net(t, x, *args, **kwargs)
        return x