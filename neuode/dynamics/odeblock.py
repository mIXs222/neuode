"""
ODE Blocks as wrapper of torchdiffeq
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from neuode.interface.common import DynamicMap
from neuode.interface.struct import ODEDMapSpec


# ode integrator block
class ODEBlock(nn.Module):

    def __init__(self, dyn_map, spec):
        super(ODEBlock, self).__init__()
        self.dyn_map = dyn_map
        self.spec = spec


    def forward(self, x, timesteps=None):
        # prepare evaluation time
        if timesteps is None:
            ts = torch.tensor([0, 1]).type_as(x)
        else:
            ts = timesteps.type_as(x)

        # pick solver
        if self.spec.use_adjoint:
            odesolver = odeint_adjoint
        else:
            odesolver = odeint

        # solve for output
        out = odesolver(
            self.dyn_map,
            x,
            ts,
            rtol=self.spec.tol,
            atol=self.spec.tol,
            method=self.spec.method,
            options={'max_num_steps': self.spec.max_num_steps})
        return out[1] if timesteps is None else out


    def trajectory(self, x, ltime, rtime, num_timesteps):
        timesteps = torch.linspace(ltime, rtime, num_timesteps)
        if ltime != 0.0:
            timesteps = torch.cat([torch.Tensor([0.0]), timesteps], 0)
        out = self.forward(x, timesteps=timesteps)
        return out[1:] if ltime != 0.0 else out


# ode block dynamic function
class ODEDMap(DynamicMap):

    def __init__(self, dyn_map, spec):
        super(DynamicMap, self).__init__()

        # check spec type
        assert isinstance(spec, ODEDMapSpec)

        # build ode block with base function
        self.net = odeblock.ODEBlock(dyn_map, spec.odeblock)
        self.use_time = spec.use_time


    def forward(self, t, x, *args, **kwargs):
        if self.use_time:
            x = util.wrap_time(t, x)
        return self.net(x)