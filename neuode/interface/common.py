"""
Common Interfaces
"""

import torch.nn as nn

# ode step function interface


class DynamicMap(nn.Module):

    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, t, x, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, t, x, *args, **kwargs):
        return self.forward(t, x, *args, **kwargs)
