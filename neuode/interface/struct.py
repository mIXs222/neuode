"""
Structs
"""

from dataclasses import dataclass
from enum import Enum 


# Activation functions
class ActivationFn(Enum):
    NONE = 0
    THRESHOLD = 1
    RELU = 2
    RRELU = 3
    HARDTANH = 4
    RELU6 = 5
    SIGMOID = 6
    TANH = 7
    ELU = 8
    SELU = 9
    GLU = 10
    HARDSHRINK = 11
    LEAKYRELU = 12
    LOGSIGMOID = 13
    SOFTPLUS = 14
    SOFTSHRINK = 15
    PRELU = 16
    SOFTSIGN = 17 
    TANHSHRINK = 18
    SOFTMIN = 19
    SOFTMAX = 20
    SOFTMAX2D = 21
    LOGSOFTMAX = 22


# Specifications for linear mapping
@dataclass
class LinearSpec:
    in_dim: int
    out_dim: int
    act_fn: ActivationFn
    use_time: bool = False


# Specifications for convolution layer
@dataclass
class ConvSpec:
    in_channel: int
    out_channel: int
    kernel_size: int
    stride: int
    padding: int
    act_fn: ActivationFn
    use_time: bool = False


# VAEEncoder specs
@dataclass
class VAESpec:
    channel: int
    height: int
    width: int
    z_dim: int
    init_std: float


# ODE Block options
@dataclass
class ODEBlockSpec:
    method: str = 'dopri5'
    use_adjoint: bool = False
    tol: float = 1e-3
    max_num_steps: int = 1000