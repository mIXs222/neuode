"""
Structs
"""

from dataclasses import dataclass
from enum import Enum 
from typing import List


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


# VAEEncoder specs
@dataclass
class VAESpec:
    channel: int
    height: int
    width: int
    z_dim: int
    init_std: float

# Divergence method spec
@dataclass
class DivSpec:
    kind: str = 'approx'  # 'approx' or 'bruteforce'
    dist: str = 'normal'  # 'normal' or 'rademacher'


# Integrator spec
@dataclass
class IntSpec:
    pass


# ODE Block options
@dataclass
class ODEBlockSpec(IntSpec):
    method: str = 'dopri5'
    use_adjoint: bool = True
    tol: float = 1e-3
    max_num_steps: int = 1000


# Basic block spec, f: X -> Y
@dataclass
class DMapSpec:
    pass


# Specifications for linear mapping
@dataclass
class LinearSpec(DMapSpec):
    in_dim: int
    out_dim: int
    act_fn: ActivationFn
    use_time: bool = False


# Specifications for convolution layer
@dataclass
class ConvSpec(DMapSpec):
    in_channel: int
    out_channel: int
    kernel_size: int
    stride: int
    padding: int
    act_fn: ActivationFn
    use_time: bool = False


# ODE dynamic function
@dataclass
class ODEDMapSpec(DMapSpec):
    odeblock: IntSpec
    base: DMapSpec
    use_time: bool = False


# Probabilistic dynamic function
@dataclass
class FFJORDProbDMapSpec(DMapSpec):
    div_fn: DivSpec
    use_time: bool = False


# ODE dynamic function
@dataclass
class SequentialSpec(DMapSpec):
    specs: List[DMapSpec]