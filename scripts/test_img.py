#!/usr/bin/env python

from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from neuode.interface.common import DynamicMap
from neuode.interface.struct import *
from neuode.dynamics.composite import build_dyn
from neuode.util.logging import logger
import neuode.util.logging as logging

DAMP_1 = 2.0
DIFFUSE = 0.05
def test_sample_1(n):
    N, M = 30, 40
    x = torch.rand(n, 3, N, M)
    y = torch.zeros(n, 3, N, M)
    y[:, 0, ...] = torch.from_numpy(cv2.blur(x[:, 0, ...].numpy(), (3, 3)))
    for i, j in product(range(N), range(M)):
            y[:, 1] = (1 - DIFFUSE) * y[:, 1] + DIFFUSE * y[:, 2]
            y[:, 2] = (1 - DIFFUSE) * y[:, 2] + DIFFUSE * y[:, 1]
    return x, y


if __name__ == '__main__':
    # pick problem
    test_sample = test_sample_1

    # build model
    cfn_specs = SequentialSpec([
        ConvSpec(3, 5, kernel_size=3, stride=1, padding=1, 
                 act_fn=ActivationFn.NONE),
        ConvSpec(5, 3, kernel_size=3, stride=1, padding=1,
                 act_fn=ActivationFn.NONE),
    ])
    ode_spec = ODEBlockSpec(use_adjoint=True)
    net = build_dyn(cfn_specs, ode_spec)

    # train
    lr, momentum = 0.001, 0.9
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(300):
        data, label = test_sample(100)
        optimizer.zero_grad()
        pred = net(data)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        logger.info('Epoch %3d: loss= %f'%(epoch, loss.item()))

        if epoch in [99, 199, 250]:
            lr /= 10
            momentum -= 0.1
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # accuracy
    with torch.no_grad():
        data, label = test_sample(1000)
        pred = net(data)
        loss = criterion(pred, label)
        logger.info('Final MSE loss= %f'%(loss.item()))

    with torch.no_grad():
        # generate initial point and find trajectory
        L, R, NT = 0.0, 1.0, 500
        NX = 1
        x0, _ = test_sample(1)
        for i, j in product(range(x0.shape[2]), range(x0.shape[3])):
            x0[0, 0, i, j] = (i-x0.shape[2]/2)**2 + (j-x0.shape[3]/2)**2
            x0[0, 1, i, j] = (i-x0.shape[2]/2)**2 + (j-x0.shape[3]/2)**2
            x0[0, 2, i, j] = (i-x0.shape[2]/2)**2 + (j-x0.shape[3]/2)**2

        traj = net.trajectory(x0, L, R, NT).numpy()

        for i in range(NX):
            logging.render_video(traj[:, i, ...], path='dump/dummy_%d.mp4'%i)