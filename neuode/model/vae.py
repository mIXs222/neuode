"""
Variational Autoencoder

from https://github.com/ldeecke/vae-torch/blob/master/architecture/nn.py
paper: https://arxiv.org/pdf/1312.6114.pdf
"""

from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from neuode.interface.common import *
from neuode.interface.struct import *


def reset_modules(net, init_std):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, init_std)
            m.bias.data.normal_(0.0, init_std)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, init_std)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, init_std)
            m.bias.data.zero_()


class VAEEncoder(nn.Module):

    def __init__(self, spec):
        super(VAEEncoder, self).__init__()
        self.conv_dim = [128, spec.height // 16, spec.width // 16]
        self.c_dim = np.prod(self.conv_dim)
        self.encode = nn.Sequential(
            nn.Conv2d(spec.channel, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.linear_mu = nn.Linear(self.c_dim, spec.z_dim)
        self.linear_log_sigma_sq = nn.Linear(self.c_dim, spec.z_dim)
        reset_modules(self, spec.init_std)


    def forward(self, x):
        h = self.encode(x)
        return self.linear_mu(h.reshape(x.size(0), -1)), \
               self.linear_log_sigma_sq(h.reshape(x.size(0), -1))


class VAEDecoder(nn.Module):

    def __init__(self, spec):
        super(VAEDecoder, self).__init__()
        self.conv_dim = [128, spec.height // 16, spec.width // 16]
        self.c_dim = np.prod(self.conv_dim)
        self.z2c = nn.Sequential(
            nn.Linear(spec.z_dim, self.c_dim),
            nn.ReLU(),
            nn.Linear(self.c_dim, self.c_dim),
            nn.ReLU(),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, spec.channel, 4, 2, 1, bias=True),
            nn.Sigmoid(),
        )
        reset_modules(self, spec.init_std)


    def forward(self, x):
        h = self.z2c(x)
        x = self.decode(h.reshape(x.size(0), *self.conv_dim))
        return x


class VAE(nn.Module):

    def __init__(self, spec):
        super(VAE, self).__init__()
        assert isinstance(spec, VAESpec)
        self.nets = nn.ModuleList([VAEEncoder(spec), VAEDecoder(spec)])


    def forward(self, x, ret_latent=False):
        mu, log_sigma_sq = self.nets[0](x)
        z = Variable(torch.randn(mu.size()), requires_grad=False)
        z = mu + torch.exp(log_sigma_sq / 2.0) * z
        y = self.nets[1](z)
        if ret_latent:
            return y, mu, log_sigma_sq
        return y


def vae_loss(vae, data):
    bce_loss = torch.nn.BCELoss(size_average=True)
    rcnst, mu, log_sigma_sq = vae(data, ret_latent=True)
    loss_r = bce_loss(rcnst, data) # / data.size(0)
    loss_kl_elm = (mu**2) + torch.exp(log_sigma_sq) - 1 - log_sigma_sq
    loss_kl = torch.mean(loss_kl_elm / 2.0)
    return 3*loss_r + loss_kl, rcnst


def generate_vae(loader, z_dim=256, nepoch=100, lr=0.01, momentum=0.9, 
                 verbose=False):
    sample, _ = next(iter(loader))
    _, channel, height, width = sample.shape
    vae_spec = VAESpec(channel, height, width, z_dim, 0.02)
    vae = VAE(vae_spec)

    vae.train()
    for epoch in range(nepoch):
        batch_total = len(loader)
        optimizer = optim.SGD(vae.parameters(), lr=lr, momentum=momentum)
        for batch_idx, (data, _) in enumerate(loader, 0):
            optimizer.zero_grad()
            loss, rcnst = vae_loss(vae, data)
            loss.backward()
            optimizer.step()
            if verbose and (batch_idx % 100 == 0 or batch_idx == batch_total-1):
                print('Epoch %3d [%4d/%4d (%2d%%)]: loss= %f'%(
                    epoch, batch_idx, batch_total, 
                    int(100 * batch_idx / batch_total), loss.item()))

        # plot result per epoch
        # import matplotlib.pyplot as plt
        # NR, NC, SZ = 2, 10, 1.5
        # fig, axes = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*SZ, NR*SZ))
        # for d, r, ax1, ax2 in zip(data, rcnst, axes[0], axes[1]):
        #     ax1.imshow(d.detach().numpy()[0])
        #     ax2.imshow(r.detach().numpy()[0])
        # for ax in axes.flatten():
        #     ax.axis('off')
        # plt.show()

        lr *= 0.75
    vae.eval()
    return vae


def save_vae(vae, path):
    torch.save(vae.state_dict(), path)


def build_vae(spec, path):
    vae = VAE(spec)
    vae.load_state_dict(torch.load(path))
    vae.eval()
    return vae

if __name__ == '__main__':
    import torchvision

    DPATH = '~/Documents/dataset'
    BATCH_TRAIN_SIZE = 64

    mnist = torchvision.datasets.MNIST(DPATH, train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=BATCH_TRAIN_SIZE, shuffle=True)
    generate_vae(mnist_loader, z_dim=64, nepoch=50, lr=0.1, verbose=True)