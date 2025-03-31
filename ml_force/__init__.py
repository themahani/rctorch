"""ML-Force: Morris-Lecar neural networks with force learning"""

from .models import LIF, MorrisLecar, MorrisLecarCurrent, minmax_transform, z_transform
from .optimizers import BruteForceMesh, CoordinateDescent, ParticleSwarmOptimizer
from .plots import plot_model
from .supervisors import HyperChaoticAttractor, LorenzAttractor, VanDerPol

__version__ = "0.1.0"
__all__ = [
    # Models
    "MorrisLecar",
    "MorrisLecarCurrent",
    "LIF",
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
    # Plotting
    "plot_model",
]
