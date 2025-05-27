"""
Base sampler class for discrete Gaussian sampling algorithms.

This module provides the abstract base class for all sampling algorithms,
including Klein's algorithm and MCMC-based samplers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingStats:
    """Statistics collected during sampling."""
    samples_generated: int = 0
    time_elapsed: float = 0.0
    acceptance_rate: Optional[float] = None
    extra_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_stats is None:
            self.extra_stats = {}


class DiscreteGaussianSampler(ABC):
    """
    Abstract base class for discrete Gaussian sampling over lattices.
    
    The discrete Gaussian distribution over lattice Λ with center c and 
    parameter σ is defined as:
    
    D_{Λ,σ,c}(x) = ρ_{σ,c}(x) / ρ_{σ,c}(Λ)
    
    where ρ_{σ,c}(x) = exp(-||x-c||²/(2σ²))
    """
    
    def __init__(self, lattice, sigma: float, center: Optional[np.ndarray] = None):
        """
        Initialize sampler.
        
        Args:
            lattice: Lattice instance
            sigma: Standard deviation parameter
            center: Center of the distribution (default: origin)
        """
        self.lattice = lattice
        self.sigma = sigma
        self.dimension = lattice.dimension
        
        if center is None:
            self.center = np.zeros(self.dimension)
        else:
            self.center = np.array(center, dtype=np.float64)
            
        if len(self.center) != self.dimension:
            raise ValueError(f"Center dimension {len(self.center)} != lattice dimension {self.dimension}")
        
        # Statistics tracking
        self.stats = SamplingStats()
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized {self.__class__.__name__} with σ={sigma:.4f}")
    
    def _validate_parameters(self):
        """Validate sampling parameters."""
        if self.sigma <= 0:
            raise ValueError(f"Standard deviation must be positive, got {self.sigma}")
        
        # Check if sigma is above smoothing parameter (recommended for security)
        smoothing_param = self.lattice.smoothing_parameter()
        if self.sigma < smoothing_param:
            logger.warning(
                f"σ={self.sigma:.4f} is below smoothing parameter "
                f"η={smoothing_param:.4f}. Sampling may be biased."
            )
    
    @abstractmethod
    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate samples from the discrete Gaussian distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, dimension) containing lattice points
        """
        pass
    
    def sample_coefficients(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate coefficient vectors x such that Bx are lattice points.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, dimension) containing integer coefficients
        """
        # Sample lattice points
        lattice_points = self.sample(num_samples)
        
        # Solve for coefficients: B * x = lattice_point
        coefficients = np.zeros((num_samples, self.dimension), dtype=int)
        for i, point in enumerate(lattice_points):
            # Use least squares to handle numerical errors
            coeffs = np.linalg.lstsq(self.lattice.basis, point, rcond=None)[0]
            coefficients[i] = np.round(coeffs).astype(int)
        
        return coefficients
    
    def gaussian_weight(self, point: np.ndarray) -> float:
        """
        Compute Gaussian weight ρ_{σ,c}(point).
        
        Args:
            point: Lattice point
            
        Returns:
            Gaussian weight exp(-||point-c||²/(2σ²))
        """
        diff = point - self.center
        return np.exp(-np.dot(diff, diff) / (2 * self.sigma**2))
    
    def log_gaussian_weight(self, point: np.ndarray) -> float:
        """
        Compute log Gaussian weight for numerical stability.
        
        Args:
            point: Lattice point
            
        Returns:
            Log Gaussian weight -||point-c||²/(2σ²)
        """
        diff = point - self.center
        return -np.dot(diff, diff) / (2 * self.sigma**2)
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.stats = SamplingStats()
    
    def get_stats(self) -> SamplingStats:
        """Get current sampling statistics."""
        return self.stats
    
    def empirical_mean(self, samples: np.ndarray) -> np.ndarray:
        """Compute empirical mean of samples."""
        return np.mean(samples, axis=0)
    
    def empirical_covariance(self, samples: np.ndarray) -> np.ndarray:
        """Compute empirical covariance of samples."""
        return np.cov(samples.T)
    
    def theoretical_covariance(self) -> np.ndarray:
        """
        Compute theoretical covariance matrix.
        
        For large σ, the discrete Gaussian approximates a continuous Gaussian
        with covariance σ²I.
        """
        return self.sigma**2 * np.eye(self.dimension)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"lattice={self.lattice.name}, "
                f"σ={self.sigma:.4f}, "
                f"center={self.center})")


class OneDimensionalSampler:
    """
    Helper class for 1D discrete Gaussian sampling over integers.
    
    This is used as a building block in Klein's algorithm.
    """
    
    @staticmethod
    def sample(mean: float, sigma: float, precision: int = 10) -> int:
        """
        Sample from 1D discrete Gaussian D_{Z,σ,μ}.
        
        Uses rejection sampling with precision parameter.
        
        Args:
            mean: Center of the distribution
            sigma: Standard deviation
            precision: Number of standard deviations to consider
            
        Returns:
            Integer sample
        """
        # Compute range to consider
        lower = int(np.floor(mean - precision * sigma))
        upper = int(np.ceil(mean + precision * sigma))
        
        # Compute probabilities
        x_range = np.arange(lower, upper + 1)
        log_probs = -(x_range - mean)**2 / (2 * sigma**2)
        
        # Normalize (in log space for stability)
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        
        # Sample
        return np.random.choice(x_range, p=probs)
    
    @staticmethod
    def sample_centered(sigma: float, precision: int = 10) -> int:
        """Sample from centered 1D discrete Gaussian D_{Z,σ,0}."""
        return OneDimensionalSampler.sample(0.0, sigma, precision)