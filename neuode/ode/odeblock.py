"""
ODE Blocks as wrapper of torchdiffeq
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


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
        assert ltime <= rtime
        timesteps = torch.linspace(ltime, rtime, num_timesteps)
        return self.forward(x, timesteps=timesteps)