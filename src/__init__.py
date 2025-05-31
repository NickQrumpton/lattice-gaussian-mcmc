"""
Lattice Gaussian MCMC Sampler Package

A Python package for efficient MCMC sampling from Gaussian distributions on lattices.
"""

__version__ = "0.1.0"
__author__ = "Nicholas Zhao"

from . import core
from . import models
from . import samplers
from . import diagnostics
from . import visualization

__all__ = ["core", "models", "samplers", "diagnostics", "visualization"]