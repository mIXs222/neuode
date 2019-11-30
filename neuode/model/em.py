"""
EM algorithms
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from neuode.interface.common import DynamicMap


class GMMTorch:

    def __init__(self, n_components, x, gmm_kwargs=dict()):
        # fit mixture model
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        bsize = x.shape[0]
        g = GaussianMixture(n_components=n_components, **gmm_kwargs) \
            .fit(x.reshape(bsize, -1))
        # g.covariances_ = [np.eye(x.shape[-1])] * bsize
        if len(g.covariances_.shape) == 2:
            g.covariances_ = np.array([np.diag(var) for var in g.covariances_])

        # extract mean and covariance to comppute pdf later
        self.log_weights = torch.Tensor(np.log(g.weights_))
        self.mnormals = [
            MultivariateNormal(
                torch.Tensor(mu),
                torch.Tensor(var)) for mu,
            var in zip(
                g.means_,
                g.covariances_)]
        self.means = torch.Tensor(g.means_)
        self.covariances = torch.Tensor(g.covariances_)


    def predict(self, x, normalize=False):
        likelihoods = torch.stack(
            [mnormal.log_prob(x.reshape(x.shape[0], -1)) + log_weight
            for mnormal, log_weight in zip(self.mnormals, self.log_weights)],
            dim=1)
        if normalize:
            norm = torch.logsumexp(likelihoods, dim=1)[:, None]
            likelihoods = likelihoods - norm
        return likelihoods


    def likelihood(self, x):
        # get likelihood on this mixture, q(c=this | x)
        # output shape: [batch,]
        likelihoods = self.predict(x)
        return torch.logsumexp(likelihoods, dim=1)


class MultiGMMTorch:

    def __init__(self, n_classes, n_components, x, y):
            # fit mixture models for each class
        self.n_classes = n_classes
        self.n_components = n_components
        self.gmms = []
        for c in range(n_classes):
            self.gmms.append(GMMTorch(n_components, x[y == c]))


    def posterior(self, x):
        # get posterior of all classes, q(c | x)
        # output shape: [batch, n_classes]
        probs = torch.stack([gmm.likelihood(x) for gmm in self.gmms], dim=1)
        norm = torch.logsumexp(probs, dim=1)
        return probs - norm[:, None].repeat(1, len(self.gmms))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import seaborn as sns

    n_classes, n_components = 3, 2
    n_samples = 2000

    # generate distributions
    mus = np.random.randn(n_classes, n_components) * 10
    sigmas = np.random.randn(n_classes, n_components) * 0.25

    # generate samples
    ys = (np.random.rand(n_samples) * n_classes).astype(int)
    zs = (np.random.rand(n_samples) * n_components).astype(int)
    xs = mus[ys, zs] + sigmas[ys, zs] * np.random.randn(n_samples)

    # fit to multi-gmm
    mgt = MultiGMMTorch(n_classes, n_components, xs, ys)
    pred_p = mgt.posterior(torch.Tensor(xs))
    for p, xi, yi, zi in zip(pred_p, xs, ys, zs):
        print(p, xi, yi, zi)
