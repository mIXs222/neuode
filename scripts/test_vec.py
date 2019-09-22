#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from neuode.interface.common import *
from neuode.interface.struct import *
from neuode.util.logging import logger
import neuode.function.linear as linear
import neuode.ode.odeblock as odeblock

def test_sample_1(n):
    x = torch.rand(n, 3)
    y = torch.zeros(n, 2)
    y[:, 0] = 3 * x[:, 0] + 4 * x[:, 1]
    y[:, 1] = 5 * x[:, 1] + 1 * x[:, 2]
    return x, y

OMEGA = 0.3
DAMP = 1.0
def test_sample_2(n):
    x = torch.randn(n, 2)
    y = torch.zeros(n, 2)
    y[:, 0] = DAMP * (np.cos(OMEGA) * x[:, 0] - np.sin(OMEGA) * x[:, 1])
    y[:, 1] = DAMP * (np.sin(OMEGA) * x[:, 0] + np.cos(OMEGA) * x[:, 1])
    # for i in range(n):
    #     pxs = [x[i, 0].item(), y[i, 0].item()]
    #     pys = [x[i, 1].item(), y[i, 1].item()]
    #     plt.plot(pxs, pys, 'x-')
    # plt.show()
    return x, y

if __name__ == '__main__':
    # pick problem
    test_sample = test_sample_2

    # build model
    lfn_specs = [
        LinearSpec(2, 4, ActivationFn.NONE),
        LinearSpec(4, 2, ActivationFn.NONE),
    ]
    ode_spec = ODEBlockSpec(use_adjoint=True)
    lfn = linear.MLPDMap(lfn_specs)
    net = odeblock.ODEBlock(lfn, ode_spec)

    # train
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(200):
        data, label = test_sample(1000)
        optimizer.zero_grad()
        pred = net(data)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        logger.info('Epoch %3d: loss= %f'%(epoch, loss.item()))

    # accuracy
    with torch.no_grad():
        data, label = test_sample(1000)
        pred = net(data)
        loss = criterion(pred, label)
        logger.info('Final MSE loss= %f'%(loss.item()))

    with torch.no_grad():
        # generate initial point and find trajectory
        L, R, NT = 0.0, 100.0, 100
        NX = 10
        x0 = torch.Tensor([
            [np.cos(2 * i * np.pi / NX), np.sin(2 * i * np.pi / NX)]
             for i in range(NX)])
        traj = net.trajectory(x0, L, R, NT).numpy()

        # plotting
        fig, ax = plt.subplots()
        for i in range(NX):
            tr = traj[:, i, :]
            ax.plot(tr[:, 0], tr[:, 1])
            # ax.scatter(tr[:, 0], tr[:, 1],
            #             c=np.linspace(0.0, 1.0, NT), cmap=cm.hot)
        ax.scatter(x0[:, 0], x0[:, 1], marker='x')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.show()
