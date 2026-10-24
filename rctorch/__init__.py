"""rcTorch: Reservoir computing solution using pyTorch"""

__version__ = "0.1.0"
__all__ = [
    # Optimizers
    "BruteForceMesh",
    "HyperChaoticAttractor",
    "LorenzAttractor",
    "ParticleSwarmOptimizer",
    # Reservoirs
    "Reservoir",
    # Supervisors
    "VanDerPol",
    # Transformations
    "minmax_transform",
    # Models sub-package
    "models",
    "z_transform",
]

from . import models
from .optimizers import BruteForceMesh, ParticleSwarmOptimizer
from .reservoir import Reservoir
from .supervisors import HyperChaoticAttractor, LorenzAttractor, VanDerPol
from .utils import minmax_transform, z_transform
