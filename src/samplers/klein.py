"""
Refined implementation of Klein's algorithm with improved numerical stability.

This version includes:
- Better handling of extreme parameters
- Optimized 1D discrete Gaussian sampling
- Numerical stability improvements
- Caching for performance
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from functools import lru_cache
from scipy.special import logsumexp
from .base import DiscreteGaussianSampler

logger = logging.getLogger(__name__)


class RefinedKleinSampler(DiscreteGaussianSampler):
    """
    Refined Klein's algorithm with numerical stability improvements.
    """
    
    def __init__(self, lattice, sigma: float, center: Optional[np.ndarray] = None,
                 precision: int = 10, use_log_space: bool = True):
        """
        Initialize refined Klein sampler.
        
        Args:
            lattice: Lattice instance
            sigma: Standard deviation parameter
            center: Center of the distribution
            precision: Precision parameter for 1D sampling
            use_log_space: Use log-space computations for stability
        """
        super().__init__(lattice, sigma, center)
        self.precision = precision
        self.use_log_space = use_log_space
        
        # Precompute QR decomposition with stability checks
        self._precompute_qr_stable()
        
        # Cache for 1D sampling
        self._sample_cache = {}
        self._max_cache_size = 10000
        
        # Validate parameters
        self._validate_klein_parameters()
        
        # Precompute constants
        self._log_2pi = np.log(2 * np.pi)
        self._log_sigma = np.log(self.sigma)
    
    def _precompute_qr_stable(self):
        """Compute QR decomposition with numerical stability checks."""
        # Standard QR decomposition
        self.Q, self.R = np.linalg.qr(self.lattice.basis, mode='full')
        # Identity permutation since we're not using pivoting
        self.P = np.arange(self.dimension)
        
        # Check conditioning
        condition_number = np.linalg.cond(self.R)
        if condition_number > 1e10:
            logger.warning(f"Poorly conditioned basis: condition number = {condition_number:.2e}")
        
        # Ensure R has positive diagonal (for consistency)
        for i in range(self.dimension):
            if self.R[i, i] < 0:
                self.R[i, :] *= -1
                self.Q[:, i] *= -1
        
        # Transform center accounting for pivoting
        self.center_transformed = self.Q.T @ self.center
        
        # Store diagonal elements for efficiency
        self.R_diag = np.diag(self.R)
        self.log_R_diag = np.log(np.abs(self.R_diag) + 1e-300)  # Avoid log(0)
    
    def _validate_klein_parameters(self):
        """Enhanced parameter validation."""
        min_gs_norm = self.lattice.min_gram_schmidt_norm
        
        # Klein's requirement: σ = Ω(√log n · max||b*||)
        klein_lower = min_gs_norm / np.sqrt(2 * np.log(self.dimension + 1))
        
        # Theoretical optimal from paper
        theoretical_optimal = min_gs_norm / (2 * np.sqrt(np.pi))
        
        if self.sigma < klein_lower * 0.9:  # Allow 10% margin
            logger.warning(
                f"σ={self.sigma:.4f} is below Klein's requirement "
                f"(minimum ≈ {klein_lower:.4f}). Sampling may fail."
            )
        
        # Check if we're in the optimal range
        if theoretical_optimal * 0.8 <= self.sigma <= theoretical_optimal * 1.2:
            logger.info(f"σ is near optimal for BDD applications")
    
    @lru_cache(maxsize=1000)
    def _compute_1d_probabilities(self, mean: float, sigma: float, 
                                 precision: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D discrete Gaussian probabilities with caching.
        
        Returns:
            Tuple of (support_points, log_probabilities)
        """
        # Adaptive range based on sigma
        if sigma < 0.1:
            # Very small sigma - use smaller range
            range_factor = max(3, precision)
        else:
            range_factor = precision
        
        # Support range
        lower = int(np.floor(mean - range_factor * sigma))
        upper = int(np.ceil(mean + range_factor * sigma))
        
        # Ensure reasonable range
        max_range = 1000
        if upper - lower > max_range:
            center = int(np.round(mean))
            lower = center - max_range // 2
            upper = center + max_range // 2
        
        support = np.arange(lower, upper + 1)
        
        if self.use_log_space:
            # Log-space computation for stability
            log_probs = -0.5 * ((support - mean) / sigma)**2
            # Normalize in log space
            log_probs = log_probs - logsumexp(log_probs)
        else:
            # Direct computation
            probs = np.exp(-0.5 * ((support - mean) / sigma)**2)
            probs = probs / np.sum(probs)
            log_probs = np.log(probs + 1e-300)
        
        return support, log_probs
    
    def _sample_1d_discrete_gaussian(self, mean: float, sigma: float) -> int:
        """
        Optimized 1D discrete Gaussian sampling.
        """
        # Round to reasonable precision for caching
        cache_key = (round(mean, 6), round(sigma, 6), self.precision)
        
        # Check cache
        if cache_key in self._sample_cache:
            support, log_probs = self._sample_cache[cache_key]
        else:
            support, log_probs = self._compute_1d_probabilities(mean, sigma, self.precision)
            
            # Cache management
            if len(self._sample_cache) < self._max_cache_size:
                self._sample_cache[cache_key] = (support, log_probs)
            else:
                # Clear cache if too large
                self._sample_cache.clear()
                self._sample_cache[cache_key] = (support, log_probs)
        
        # Sample
        if self.use_log_space:
            probs = np.exp(log_probs)
        else:
            probs = np.exp(log_probs)
        
        # Handle numerical issues
        probs = probs / np.sum(probs)  # Renormalize
        
        # Sample with fallback
        try:
            return np.random.choice(support, p=probs)
        except ValueError:
            # Fallback to uniform around mean if probabilities are corrupted
            logger.warning("Sampling fallback activated")
            return int(np.round(mean))
    
    def sample_single(self) -> np.ndarray:
        """
        Generate a single sample with improved stability.
        """
        # Initialize coefficient vector
        x = np.zeros(self.dimension, dtype=int)
        
        # Backward sampling with pivoting adjustment
        for i in range(self.dimension - 1, -1, -1):
            # Compute conditional mean
            conditional_sum = 0.0
            for j in range(i + 1, self.dimension):
                conditional_sum += self.R[i, j] * x[j]
            
            conditional_mean = (self.center_transformed[i] - conditional_sum) / self.R_diag[i]
            
            # Compute conditional standard deviation
            conditional_sigma = self.sigma / abs(self.R_diag[i])
            
            # Check for extreme values
            if conditional_sigma < 1e-10:
                # Extremely small sigma - use rounding
                x[i] = int(np.round(conditional_mean))
                logger.debug(f"Used rounding for coordinate {i} (sigma too small)")
            elif conditional_sigma > 1e10:
                # Extremely large sigma - warn and use large range
                logger.warning(f"Very large conditional sigma at coordinate {i}: {conditional_sigma:.2e}")
                x[i] = self._sample_1d_discrete_gaussian(conditional_mean, min(conditional_sigma, 1e6))
            else:
                # Normal sampling
                x[i] = self._sample_1d_discrete_gaussian(conditional_mean, conditional_sigma)
        
        # Apply inverse permutation
        x_permuted = np.zeros_like(x)
        x_permuted[self.P] = x
        
        # Transform back to lattice point
        lattice_point = self.lattice.basis @ x_permuted
        
        return lattice_point
    
    def compute_log_density(self, lattice_point: np.ndarray) -> float:
        """
        Compute log density under Klein's distribution.
        
        More accurate than the base implementation.
        """
        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(self.lattice.basis, lattice_point)
            coeffs_int = np.round(coeffs).astype(int)
            
            # Check if it's actually a lattice point
            reconstruction_error = np.linalg.norm(
                self.lattice.basis @ coeffs_int - lattice_point
            )
            if reconstruction_error > 1e-10:
                return -np.inf
        except np.linalg.LinAlgError:
            return -np.inf
        
        # Apply permutation
        coeffs_permuted = coeffs_int[self.P]
        
        # Compute log probability
        log_prob = 0.0
        
        for i in range(self.dimension - 1, -1, -1):
            # Conditional mean
            conditional_sum = sum(
                self.R[i, j] * coeffs_permuted[j] 
                for j in range(i + 1, self.dimension)
            )
            conditional_mean = (self.center_transformed[i] - conditional_sum) / self.R_diag[i]
            conditional_sigma = self.sigma / abs(self.R_diag[i])
            
            # Log probability of this coordinate
            residual = coeffs_permuted[i] - conditional_mean
            log_prob += -0.5 * (residual / conditional_sigma)**2
            
            # Log normalization (approximate)
            log_prob -= 0.5 * self._log_2pi + np.log(conditional_sigma)
            
            # Account for discrete normalization
            support, log_probs = self._compute_1d_probabilities(
                conditional_mean, conditional_sigma, self.precision
            )
            log_Z = logsumexp(log_probs)
            log_prob -= log_Z
        
        return log_prob
    
    def adaptive_precision_sample(self, target_distance: Optional[float] = None) -> np.ndarray:
        """
        Sample with adaptive precision based on target accuracy.
        
        Args:
            target_distance: Target distance from center (for BDD)
            
        Returns:
            Lattice point
        """
        if target_distance is None:
            # Use default precision
            return self.sample_single()
        
        # Adapt precision based on target
        old_precision = self.precision
        
        # Higher precision for closer targets
        relative_distance = target_distance / self.sigma
        if relative_distance < 1:
            self.precision = max(20, 2 * old_precision)
        elif relative_distance > 5:
            self.precision = max(5, old_precision // 2)
        
        try:
            sample = self.sample_single()
        finally:
            self.precision = old_precision
        
        return sample
    
    def parallel_sample_batch(self, num_samples: int, batch_size: int = 100) -> np.ndarray:
        """
        Generate samples in batches for better cache utilization.
        """
        samples = []
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_samples = [
                self.sample_single() 
                for _ in range(batch_end - i)
            ]
            samples.extend(batch_samples)
            
            # Clear cache periodically to avoid memory issues
            if len(self._sample_cache) > self._max_cache_size:
                self._sample_cache.clear()
        
        return np.array(samples)
    
    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate multiple samples from the discrete Gaussian distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, dimension) containing lattice points
        """
        if num_samples == 1:
            return self.sample_single().reshape(1, -1)
        else:
            return self.parallel_sample_batch(num_samples)
    
    def diagnostic_info(self) -> Dict[str, Any]:
        """
        Return diagnostic information about the sampler.
        """
        return {
            'algorithm': 'Refined Klein',
            'sigma': self.sigma,
            'precision': self.precision,
            'use_log_space': self.use_log_space,
            'cache_size': len(self._sample_cache),
            'condition_number': np.linalg.cond(self.R),
            'min_R_diag': np.min(np.abs(self.R_diag)),
            'max_R_diag': np.max(np.abs(self.R_diag)),
            'min_conditional_sigma': self.sigma / np.max(np.abs(self.R_diag)),
            'max_conditional_sigma': self.sigma / np.min(np.abs(self.R_diag)),
        }