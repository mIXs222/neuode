"""
FFJORD dynamics to estimate density p(x) from tractable p(z)

paper -> https://arxiv.org/abs/1810.01367
from https://github.com/rtqichen/ffjord/blob/master
"""

from functools import partial

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from neuode.interface.common import DynamicMap
from neuode.interface.struct import FFJORDProbDMapSpec, ODEBlockSpec, DivSpec


# Tr(df/dx(t)), divergence by bruteforce
def div_bruteforce(f, xt, *unused_args, **unused_kwargs):
    sum_diag = 0.
    for i in range(xt.shape[1]):
        f_i = f[:, i].sum()
        sum_diag += torch.autograd.grad(f_i, xt, create_graph=True)[0] \
            .contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def draw_random_dir(xt, dist='normal'):
    # sample e ~ X s.t. E[X] = 0, Cov(X) = I
    # TODO: sample more than one?
    if dist == 'rademacher':
        # X = {-1, 1}^n
        e = torch.randint(low=0, high=2, size=xt.shape).to(xt) * 2 - 1
    elif dist == 'normal':
        # X = N(0, I_n)
        e = torch.randn_like(xt)
    else:
        raise ValueError('distribution not supported for hutchinson')
    return e


# Tr(df/dx(t)) ~ E[e^T df/dx(t) e], approximated divergence
#   dist: 'normal', 'rademacher'
def div_hutchinson(f, xt, e):
    e_dfdxt = torch.autograd.grad(f, xt, e, create_graph=True)[0]
    e_dfdxt_e = e_dfdxt * e
    div = e_dfdxt_e.view(xt.shape[0], -1).sum(dim=1)
    return div


def build_div_fn(spec):
    assert isinstance(spec, DivSpec)
    if spec.kind == 'bruteforce':
        return div_bruteforce
    elif spec.kind == 'approx':
        return partial(div_hutchinson)
    raise ValueError('Given kind of div fn not supported')


# augmented dynamic function preserving density
class FFJORDProbDMap(DynamicMap):

    def __init__(self, dyn_map, spec):
        super(DynamicMap, self).__init__()

        # check spec type
        assert isinstance(spec, FFJORDProbDMapSpec)

        # save necessary construct div method
        self.dyn_map = dyn_map
        self.div_fn = build_div_fn(spec.div_fn)
        if spec.div_fn.dist is None:
            self.draw_e = None
        else:
            self.draw_e = partial(draw_random_dir, dist=spec.div_fn.dist)
        self.e = None

    def forward(self, t, xlogpx, *args, **kwargs):
        # augmented dyn fn, f_aug in Algorithm 1 on page 5
        xt, logpx = xlogpx
        if self.e is None or self.e.shape != xt.shape:
            # redraw epsilon since the shapes don't match
            self.e = None if self.draw_e is None else self.draw_e(xt)
        with torch.set_grad_enabled(True):
            xt.requires_grad_(True)
            f = self.dyn_map(t, xt)
            div = self.div_fn(f, xt, self.e).view(xt.shape[0], 1)
        return (f, -div)

    def reset(self):
        self.e = None


# ode integrator block
class FFJORDBlock(nn.Module):

    def __init__(self, probdyn_map, spec, pdf_z=None):
        # provide pdf_z to enable adjustment by logpz - delta_logpx
        super(FFJORDBlock, self).__init__()

        # sanity check
        assert isinstance(spec, ODEBlockSpec)

        self.probdyn_map = probdyn_map
        self.spec = spec
        self.pdf_z = pdf_z

    def forward(self, x, timesteps=None, ret_z=False):
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

        # initialize log p(x)
        logpx = torch.zeros(x.shape[0], 1).to(x)

        # reset probdyn_map state
        self.probdyn_map.reset()

        # solve for output
        out = odesolver(
            self.probdyn_map,
            (x, logpx),
            ts,
            rtol=self.spec.tol,
            atol=self.spec.tol,
            method=self.spec.method,
            options={'max_num_steps': self.spec.max_num_steps})

        # compute posterior
        zt, delta_logpx = out
        if self.pdf_z:
            delta_logpx = self.pdf_z(zt) - delta_logpx

        if timesteps is None:
            zt = zt[1]
            delta_logpx = delta_logpx[1]
        if ret_z:
            return zt, delta_logpx
        return delta_logpx

    def trajectory(self, x, ltime, rtime, num_timesteps):
        timesteps = torch.linspace(ltime, rtime, num_timesteps)
        if ltime != 0.0:
            timesteps = torch.cat([torch.Tensor([0.0]), timesteps], 0)
        out = self.forward(x, timesteps=timesteps)
        return out[1:] if ltime != 0.0 else out
