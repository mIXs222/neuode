"""
Function compositions
"""

import torch.nn as nn

from neuode.interface.common import DynamicMap
from neuode.interface.struct import (
    IntSpec, ODEBlockSpec,
    DMapSpec, LinearSpec, ConvSpec, ODEDMapSpec, SequentialSpec,
)

from neuode.dynamics.linear import LinearDMap
from neuode.dynamics.conv import ConvDMap
from neuode.dynamics.odeblock import ODEBlock, ODEDMap


# from dynamic map spec to linear function
def build_dmap(spec):
    assert isinstance(spec, DMapSpec)
    if isinstance(spec, LinearSpec):
        return LinearDMap(spec)
    elif isinstance(spec, ConvSpec):
        return ConvDMap(spec)
    elif isinstance(spec, ODEDMapSpec):
        dmap = build_dmap(spec.odeblock)
        return ODEDMap(dmap, spec)
    elif isinstance(spec, SequentialSpec):
        return SequentialDMap(spec)
    raise ValueError('Given dynmap spec not supported')


# wrap dynamic function in integrator
def build_dyn(dmap_spec, int_spec):
    assert isinstance(int_spec, IntSpec)
    dmap = build_dmap(dmap_spec)
    if isinstance(int_spec, ODEBlockSpec):
        return ODEBlock(dmap, int_spec)
    raise ValueError('Given integrtor spec not supported')


# sequential blocks
class SequentialDMap(DynamicMap):

    def __init__(self, spec):
        super(DynamicMap, self).__init__()

        # sanity check
        assert isinstance(spec, SequentialSpec)
        assert len(spec.specs) > 0

        # construct each linear layer
        self.nets = nn.ModuleList([build_dmap(sp) for sp in spec.specs])


    def forward(self, t, x, *args, **kwargs):
        for net in self.nets:
            x = net(t, x, *args, **kwargs)
        return x


# sequential blocks
class SequentialListDMap(DynamicMap):

    def __init__(self, dyn_maps):
        super(DynamicMap, self).__init__()

        # construct each linear layer
        self.nets = nn.ModuleList(dyn_maps)


    def forward(self, t, x, *args, **kwargs):
        for net in self.nets:
            x = net(t, x, *args, **kwargs)
        return x