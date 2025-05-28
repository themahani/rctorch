"""rcTorch: Reservoir computing solution using pyTorch"""

__version__ = "0.1.0"
__all__ = [
    # Models sub-package
    "models",
    # Reservoirs
    "Reservoir",
    # Transformations
    "minmax_transform",
    "z_transform",
    # Supervisors
    "VanDerPol",
    "LorenzAttractor",
    "HyperChaoticAttractor",
    # Optimizers
    "CoordinateDescent",
    "BruteForceMesh",
    "ParticleSwarmOptimizer",
]

from . import models
from .optimizers import BruteForceMesh, CoordinateDescent, ParticleSwarmOptimizer
from .reservoir import Reservoir
from .supervisors import HyperChaoticAttractor, LorenzAttractor, VanDerPol
from .utils import minmax_transform, z_transform
