#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from neuode.interface.struct import (
    ODEBlockSpec, LinearSpec, SequentialSpec, FFJORDProbDMapSpec,
    ActivationFn, DivSpec,
)
from neuode.dynamics.linear import ConcatSquashLinear
from neuode.dynamics.composite import build_dmap, SequentialListDMap
from neuode.dynamics.ffjord_block import FFJORDProbDMap, FFJORDBlock
from neuode.util.util import log_normal_pdf, actfn2nn
from neuode.util.logging import logger


# log pdf of unit normal dist
def log_normal_pdf_2(z):
    z = z[..., :2]
    pzs = -0.5 * np.log(2 * np.pi) - z.pow(2) / 2
    return pzs.sum(-1, keepdim=True)


AUG_DIM = 0
def test_sample_1(n):
    mus = np.array([[2, 2], [-1, -2], [3, -1]])
    sigmas = np.array([0.4, 0.5, 1])
    # mus = np.array([[0, 0], [0, 0], [0, 0]])
    # sigmas = np.array([10.0, 5.0, 1.0])
    ys = np.random.randint(0, 3, size=n)
    zs = np.random.randn(n, 2)
    xs = zs * sigmas[np.tile(ys[:, None], (1, 2))] + mus[ys]
    xs = np.hstack((xs, np.zeros((n, AUG_DIM))))
    ps = np.exp(-np.sum(zs**2, axis=1))
    return torch.Tensor(xs), torch.Tensor(ys), ps


if __name__ == '__main__':
    # pick problem
    test_sample = test_sample_1

    # build model
    lfn_aug_spec = FFJORDProbDMapSpec(
        DivSpec(kind='bruteforce'),
        # DivSpec(kind='approx', dist='normal'),
        # DivSpec(kind='approx', dist='rademacher'),
    )
    ffjord_spec = ODEBlockSpec(use_adjoint=True)

    # lfn = build_dmap(lfn_specs)
    lfn = SequentialListDMap([
        ConcatSquashLinear(2+AUG_DIM, 64, ActivationFn.TANH),
        ConcatSquashLinear(64, 64, ActivationFn.TANH),
        ConcatSquashLinear(64, 64, ActivationFn.TANH),
        ConcatSquashLinear(64, 2+AUG_DIM, ActivationFn.NONE),
    ])
    lfn_aug = FFJORDProbDMap(lfn, lfn_aug_spec)
    net = FFJORDBlock(lfn_aug, ffjord_spec, pdf_z=log_normal_pdf)

    # train
    NEPOCH = 300
    LR = 0.1
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    for epoch in range(NEPOCH):
        data, _, _ = test_sample(1000)
        optimizer.zero_grad()
        logpx = net(data)
        loss = -logpx.mean()
        loss.backward()
        optimizer.step()
        logger.info('Epoch %3d: loss= %f' % (epoch, loss.item()))
        # if epoch == 2 * NEPOCH // 3:
        #     optimizer = optim.SGD(net.parameters(), lr=LR/3, momentum=0.9)

    # plot dynamic
    with torch.no_grad():
        # sample data
        xs, ys, true_p = test_sample(4000)
        true_p = -true_p
        with torch.no_grad():
            z0, logpx = net(xs, ret_z=True)
        xs = xs.detach().numpy()
        ys = ys.detach().numpy()
        z0 = z0.detach().numpy()
        logpx = logpx.detach().numpy()
        px = -np.exp(logpx[:, 0])

        # plotting
        cm = plt.cm.get_cmap('plasma')
        fig, axes = plt.subplots(ncols=3, figsize=(7.5, 2.5))
        axes[0].scatter(xs[:, 0], xs[:, 1], s=0.7, alpha=0.3, c=true_p, cmap=cm)
        axes[1].scatter(z0[:, 0], z0[:, 1], s=0.7, alpha=0.3, c=px, cmap=cm)
        axes[2].scatter(xs[:, 0], xs[:, 1], s=0.7, alpha=0.3, c=px, cmap=cm)
        plt.show()
