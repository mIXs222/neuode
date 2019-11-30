"""
Linear dynamic mappings
"""

import torch
import torch.nn as nn

from neuode.interface.common import DynamicMap
from neuode.interface.struct import LinearSpec, ActivationFn
import neuode.util.util as util


# Fully feed forward layer
class LinearDMap(DynamicMap):

    def __init__(self, spec):
        super(LinearDMap, self).__init__()

        # sanity check
        assert isinstance(spec, LinearSpec)

        # time dependent
        self.use_time = spec.use_time
        in_dim_add = 1 if self.use_time else 0

        # parse spec into linear layer
        nets = []
        nets.append(nn.Linear(spec.in_dim + in_dim_add, spec.out_dim, 
                              bias=spec.bias))
        nets.append(util.actfn2nn(spec.act_fn))
        self.net = nn.Sequential(*nets)

    def forward(self, t, x, *args, **kwargs):
        if self.use_time:
            x = util.wrap_time_vec(t, x)
        return self.net(x)


# from rtqichen/ffjord/blob/master/lib/layers/diffeq_layers/basic.py
class ConcatSquashLinear(DynamicMap):

    def __init__(self, dim_in, dim_out, actfn):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)
        self._actfn = util.actfn2nn(actfn)

    def forward(self, t, x, *args, **kwargs):
        x = self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))
        return self._actfn(x)


# from rtqichen/ffjord/blob/master/lib/layers/diffeq_layers/resnet.py
class BasicResNetBlock(DynamicMap):

    def __init__(self, dim, map_block, actfn, ngroups=16):
        super(BasicResNetBlock, self).__init__()

        if ngroups > 1:
            self._norm1 = nn.GroupNorm(ngroups, dim, eps=1e-4)
        else:
            self._norm1 = util.actfn2nn(ActivationFn.NONE)
        self._relu1 = nn.ReLU()
        self._map1 = map_block
        self._actfn = util.actfn2nn(actfn)


    def forward(self, t, x, *args, **kwargs):
        out = self._norm1(x)
        out = self._relu1(out)
        out = self._map1(t, out, *args, **kwargs)
        out = out + x
        return self._actfn(out)