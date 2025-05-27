"""Statistical models for lattice Gaussian MCMC."""

from .gmrf import GaussianMarkovRandomField
from .car import ConditionalAutoregressive
from .ising import IsingModel

__all__ = ["GaussianMarkovRandomField", "ConditionalAutoregressive", "IsingModel"]