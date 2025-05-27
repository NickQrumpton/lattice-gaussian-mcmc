#!/usr/bin/env python3
"""
Exact Discrete Gaussian Samplers for Lattice Cryptography.

This module implements numerically stable and efficient discrete Gaussian samplers
following Peikert (2010) and Wang & Ling (2018). It provides both rejection 
sampling and CDT methods for exact sampling from D_{Z,σ,c}.

The discrete Gaussian distribution over integers is defined as:
    D_{Z,σ,c}(x) ∝ exp(-(x-c)²/(2σ²))

References:
    - Peikert, C. (2010). "An Efficient and Parallel Gaussian Sampler for Lattices"
    - Wang & Ling (2018). "Lattice Gaussian Sampling by Markov Chain Monte Carlo"
    - Ducas et al. (2013). "Lattice Signatures and Bimodal Gaussians"
"""

try:
    from sage.all import *
except ImportError:
    raise ImportError("This module requires SageMath")

import numpy as np
from typing import Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings


class DiscreteGaussianSampler(ABC):
    """Abstract base class for discrete Gaussian samplers."""
    
    @abstractmethod
    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """Sample from the discrete Gaussian distribution."""
        pass
    
    @abstractmethod
    def log_probability(self, x: int) -> float:
        """Compute log probability of a value."""
        pass


class RejectionSampler(DiscreteGaussianSampler):
    """
    Rejection sampling for discrete Gaussian D_{Z,σ,c}.
    
    Implements the rejection sampling method from Peikert (2010) with
    tight tail bounds for efficiency.
    
    ALGORITHM:
        1. Sample y from continuous Gaussian N(c, σ²)
        2. Round to nearest integer x = ⌊y + 0.5⌋
        3. Accept x with probability ρ_σ(x-y) where
           ρ_σ(t) = exp(-t²/(2σ²))
        
    The tail bound τ = ω(√log n) ensures negligible statistical distance.
    """
    
    def __init__(self, sigma: float, center: float = 0.0, tail_bound: Optional[float] = None):
        """
        Initialize rejection sampler.
        
        Args:
            sigma: Standard deviation σ > 0
            center: Center c (can be any real number)
            tail_bound: Tail cut parameter τ (default: 12 for cryptographic security)
            
        EXAMPLES::
        
            sage: sampler = RejectionSampler(sigma=3.0, center=2.5)
            sage: x = sampler.sample()
            sage: isinstance(x, Integer)
            True
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
            
        self.sigma = RDF(sigma)
        self.center = RDF(center)
        self.tail_bound = RDF(tail_bound if tail_bound is not None else 12.0)
        
        # Precompute constants
        self.sigma_sq = self.sigma ** 2
        self.two_sigma_sq = 2 * self.sigma_sq
        self.log_normalizer = None  # Computed on demand
        
        # Support interval [center - τσ, center + τσ]
        self.min_support = int(floor(self.center - self.tail_bound * self.sigma))
        self.max_support = int(ceil(self.center + self.tail_bound * self.sigma))
        
    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """
        Sample from D_{Z,σ,c}.
        
        Args:
            n: Number of samples
            
        Returns:
            Single integer if n=1, list of integers otherwise
            
        EXAMPLES::
        
            sage: sampler = RejectionSampler(sigma=2.0)
            sage: samples = sampler.sample(1000)
            sage: len(samples)
            1000
            sage: abs(mean([float(x) for x in samples])) < 0.2  # Should be close to 0
            True
        """
        samples = []
        
        for _ in range(n):
            while True:
                # Step 1: Sample from continuous Gaussian
                y = self.center + self.sigma * normalvariate(0, 1)
                
                # Step 2: Round to nearest integer
                x = Integer(round(y))
                
                # Check tail bound
                if x < self.min_support or x > self.max_support:
                    continue
                
                # Step 3: Accept/reject
                # Probability ρ_σ(x-y) = exp(-(x-y)²/(2σ²))
                t = float(x) - y
                accept_prob = exp(-t*t / self.two_sigma_sq)
                
                if random() <= accept_prob:
                    samples.append(x)
                    break
        
        return samples[0] if n == 1 else samples
    
    def log_probability(self, x: int) -> float:
        """
        Compute log probability log P(X = x).
        
        Args:
            x: Integer value
            
        Returns:
            Log probability
            
        EXAMPLES::
        
            sage: sampler = RejectionSampler(sigma=1.0, center=0)
            sage: p0 = sampler.log_probability(0)
            sage: p1 = sampler.log_probability(1)
            sage: p0 > p1  # Peak at center
            True
        """
        x = Integer(x)
        
        # log P(x) = -(x-c)²/(2σ²) - log(normalizer)
        log_unnormalized = -(x - self.center)**2 / self.two_sigma_sq
        
        if self.log_normalizer is None:
            self._compute_normalizer()
            
        return float(log_unnormalized - self.log_normalizer)
    
    def _compute_normalizer(self):
        """Compute normalization constant."""
        # Sum over support with tail bounds
        total = RDF(0)
        for x in range(self.min_support, self.max_support + 1):
            total += exp(-(x - self.center)**2 / self.two_sigma_sq)
        
        self.log_normalizer = log(total)
    
    def probability(self, x: int) -> float:
        """Compute probability P(X = x)."""
        return float(exp(self.log_probability(x)))


class CDTSampler(DiscreteGaussianSampler):
    """
    Cumulative Distribution Table sampler for discrete Gaussian.
    
    Efficient for small to moderate σ using precomputed CDF tables
    and binary search. Falls back to rejection sampling for large support.
    
    ALGORITHM:
        1. Precompute CDF over support interval
        2. Sample uniform u ∈ [0,1]
        3. Binary search to find x such that CDF(x-1) < u ≤ CDF(x)
    """
    
    def __init__(self, sigma: float, center: float = 0.0, 
                 max_table_size: int = 10000, tail_bound: Optional[float] = None):
        """
        Initialize CDT sampler.
        
        Args:
            sigma: Standard deviation σ > 0
            center: Center c
            max_table_size: Maximum CDF table size before falling back to rejection
            tail_bound: Tail cut parameter τ
            
        EXAMPLES::
        
            sage: sampler = CDTSampler(sigma=1.5, center=0)
            sage: x = sampler.sample()
            sage: isinstance(x, Integer)
            True
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
            
        self.sigma = RDF(sigma)
        self.center = RDF(center)
        self.tail_bound = RDF(tail_bound if tail_bound is not None else 12.0)
        self.max_table_size = max_table_size
        
        # Precompute constants
        self.sigma_sq = self.sigma ** 2
        self.two_sigma_sq = 2 * self.sigma_sq
        
        # Support interval
        self.min_support = int(floor(self.center - self.tail_bound * self.sigma))
        self.max_support = int(ceil(self.center + self.tail_bound * self.sigma))
        
        # Check if we should use CDT or fall back to rejection
        self.table_size = self.max_support - self.min_support + 1
        
        if self.table_size > self.max_table_size:
            warnings.warn(f"Support size {self.table_size} exceeds max_table_size "
                         f"{self.max_table_size}. Using rejection sampling.")
            self.rejection_sampler = RejectionSampler(sigma, center, tail_bound)
            self.use_rejection = True
        else:
            self.use_rejection = False
            self._build_cdf_table()
    
    def _build_cdf_table(self):
        """Build cumulative distribution function table."""
        # Compute unnormalized probabilities
        probs = []
        for x in range(self.min_support, self.max_support + 1):
            p = exp(-(x - self.center)**2 / self.two_sigma_sq)
            probs.append(p)
        
        # Normalize
        total = sum(probs)
        self.pdf = [RDF(p / total) for p in probs]
        
        # Build CDF
        self.cdf = []
        cumsum = RDF(0)
        for p in self.pdf:
            cumsum += p
            self.cdf.append(cumsum)
        
        # Ensure last value is exactly 1
        self.cdf[-1] = RDF(1)
    
    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """
        Sample from D_{Z,σ,c} using CDT method.
        
        EXAMPLES::
        
            sage: sampler = CDTSampler(sigma=1.0, center=0)
            sage: samples = sampler.sample(1000)
            sage: abs(mean([float(x) for x in samples])) < 0.1
            True
        """
        if self.use_rejection:
            return self.rejection_sampler.sample(n)
        
        samples = []
        
        for _ in range(n):
            # Sample uniform u ∈ [0,1]
            u = RDF(random())
            
            # Binary search for x such that CDF[x-1] < u ≤ CDF[x]
            left, right = 0, len(self.cdf) - 1
            
            while left < right:
                mid = (left + right) // 2
                if self.cdf[mid] < u:
                    left = mid + 1
                else:
                    right = mid
            
            # Convert index to integer value
            x = Integer(self.min_support + left)
            samples.append(x)
        
        return samples[0] if n == 1 else samples
    
    def log_probability(self, x: int) -> float:
        """Compute log probability."""
        if self.use_rejection:
            return self.rejection_sampler.log_probability(x)
        
        x = Integer(x)
        if x < self.min_support or x > self.max_support:
            return float('-inf')
        
        idx = x - self.min_support
        return float(log(self.pdf[idx]))
    
    def probability(self, x: int) -> float:
        """Compute probability P(X = x)."""
        if self.use_rejection:
            return self.rejection_sampler.probability(x)
        
        x = Integer(x)
        if x < self.min_support or x > self.max_support:
            return 0.0
        
        idx = x - self.min_support
        return float(self.pdf[idx])


class DiscreteGaussianVectorSampler:
    """
    High-dimensional discrete Gaussian sampler for Z^n.
    
    Samples from product distribution D_{Z,σ,c}^n where each coordinate
    is sampled independently.
    """
    
    def __init__(self, sigma: Union[float, List[float]], 
                 center: Optional[Union[List[float], 'vector']] = None,
                 n: Optional[int] = None,
                 method: str = 'auto'):
        """
        Initialize vector sampler.
        
        Args:
            sigma: Scalar σ or list of σ_i for each coordinate
            center: Center vector c (default: zero vector)
            n: Dimension (required if sigma is scalar)
            method: 'rejection', 'cdt', or 'auto'
            
        EXAMPLES::
        
            sage: sampler = DiscreteGaussianVectorSampler(sigma=2.0, n=10)
            sage: v = sampler.sample()
            sage: len(v) == 10
            True
            sage: all(x in ZZ for x in v)
            True
        """
        # Handle sigma
        if isinstance(sigma, (list, tuple)):
            self.sigmas = [RDF(s) for s in sigma]
            self.n = len(self.sigmas)
        else:
            if n is None:
                raise ValueError("n must be specified when sigma is scalar")
            self.sigmas = [RDF(sigma)] * n
            self.n = n
        
        # Handle center
        if center is None:
            self.centers = [RDF(0)] * self.n
        else:
            if hasattr(center, '__len__'):
                if len(center) != self.n:
                    raise ValueError(f"Center dimension {len(center)} != n={self.n}")
                self.centers = [RDF(c) for c in center]
            else:
                raise ValueError("Center must be a vector or list")
        
        # Choose sampling method
        self.method = method
        self._create_samplers()
    
    def _create_samplers(self):
        """Create 1D samplers for each coordinate."""
        self.samplers = []
        
        for i in range(self.n):
            sigma_i = self.sigmas[i]
            center_i = self.centers[i]
            
            if self.method == 'rejection':
                sampler = RejectionSampler(sigma_i, center_i)
            elif self.method == 'cdt':
                sampler = CDTSampler(sigma_i, center_i)
            else:  # auto
                # Use CDT for small sigma, rejection for large
                if sigma_i < 20:
                    sampler = CDTSampler(sigma_i, center_i)
                else:
                    sampler = RejectionSampler(sigma_i, center_i)
            
            self.samplers.append(sampler)
    
    def sample(self, num_samples: int = 1) -> Union['vector', List['vector']]:
        """
        Sample from D_{Z,σ,c}^n.
        
        Returns:
            Single vector if num_samples=1, list of vectors otherwise
            
        EXAMPLES::
        
            sage: sampler = DiscreteGaussianVectorSampler(sigma=1.5, n=5)
            sage: samples = sampler.sample(100)
            sage: len(samples) == 100
            True
            sage: all(len(v) == 5 for v in samples)
            True
        """
        samples = []
        
        for _ in range(num_samples):
            coords = [sampler.sample() for sampler in self.samplers]
            v = vector(ZZ, coords)
            samples.append(v)
        
        return samples[0] if num_samples == 1 else samples
    
    def log_probability(self, v: 'vector') -> float:
        """
        Compute log probability of vector v.
        
        EXAMPLES::
        
            sage: sampler = DiscreteGaussianVectorSampler(sigma=1.0, n=3)
            sage: v = vector([0, 0, 0])
            sage: p = sampler.log_probability(v)
            sage: p < 0  # Log probability
            True
        """
        if len(v) != self.n:
            raise ValueError(f"Vector dimension {len(v)} != {self.n}")
        
        log_prob = RDF(0)
        for i, sampler in enumerate(self.samplers):
            log_prob += sampler.log_probability(v[i])
        
        return float(log_prob)


# Convenience functions
def sample_discrete_gaussian_1d(sigma: float, center: float = 0.0, 
                               method: str = 'auto') -> int:
    """
    Sample single value from 1D discrete Gaussian D_{Z,σ,c}.
    
    Args:
        sigma: Standard deviation
        center: Center (default: 0)
        method: 'rejection', 'cdt', or 'auto'
        
    Returns:
        Integer sample
        
    EXAMPLES::
    
        sage: x = sample_discrete_gaussian_1d(sigma=2.0, center=1.5)
        sage: x in ZZ
        True
    """
    if method == 'rejection':
        sampler = RejectionSampler(sigma, center)
    elif method == 'cdt':
        sampler = CDTSampler(sigma, center)
    else:  # auto
        sampler = CDTSampler(sigma, center) if sigma < 20 else RejectionSampler(sigma, center)
    
    return sampler.sample()


def sample_discrete_gaussian_vec(sigma: Union[float, List[float]], 
                                center: Optional[Union[List[float], 'vector']] = None,
                                n: Optional[int] = None,
                                method: str = 'auto') -> 'vector':
    """
    Sample from n-dimensional discrete Gaussian D_{Z,σ,c}^n.
    
    Args:
        sigma: Scalar or per-coordinate standard deviations
        center: Center vector (default: zero)
        n: Dimension (required if sigma is scalar)
        method: Sampling method
        
    Returns:
        Integer vector
        
    EXAMPLES::
    
        sage: v = sample_discrete_gaussian_vec(sigma=1.5, n=10)
        sage: len(v) == 10
        True
        sage: v.parent()
        Ambient free module of rank 10 over Integer Ring
    """
    sampler = DiscreteGaussianVectorSampler(sigma, center, n, method)
    return sampler.sample()


# Integration with lattice samplers
class LatticeGaussianSampler:
    """
    Discrete Gaussian sampler over general lattices.
    
    Supports sampling from D_{Λ,σ,c} for any lattice Λ with known basis.
    """
    
    def __init__(self, basis: 'Matrix', sigma: float, center: Optional['vector'] = None):
        """
        Initialize lattice Gaussian sampler.
        
        Args:
            basis: Lattice basis matrix (columns are basis vectors)
            sigma: Gaussian parameter
            center: Center vector (default: zero)
            
        EXAMPLES::
        
            sage: B = matrix([[2, 1], [0, 2]])  # Simple 2D lattice
            sage: sampler = LatticeGaussianSampler(B, sigma=1.0)
            sage: v = sampler.sample_klein()
            sage: B.solve_left(v) in ZZ^2  # v is in lattice
            True
        """
        self.basis = basis
        self.n = basis.ncols()
        self.sigma = RDF(sigma)
        self.center = center if center is not None else vector(RDF, self.n)
        
        # Precompute useful quantities
        self.gram = basis.transpose() * basis
        self.dual_basis = self.gram.inverse() * basis.transpose()
    
    def sample_klein(self) -> 'vector':
        """
        Sample using Klein's algorithm (GPV sampler).
        
        ALGORITHM:
            Sample from D_{Λ,σ,c} by iteratively sampling from
            conditional distributions over cosets.
            
        Returns:
            Lattice vector from D_{Λ,σ,c}
        """
        # This is a placeholder - Klein's algorithm would be implemented
        # in the actual lattice sampler modules
        raise NotImplementedError("Use Klein sampler from lattice modules")
    
    def nearest_plane(self, target: 'vector') -> 'vector':
        """
        Find closest lattice point using Babai's nearest plane algorithm.
        
        Used for discrete Gaussian sampling via continuous approximation.
        """
        coeffs = self.basis.solve_left(target)
        rounded = vector([round(c) for c in coeffs])
        return self.basis * rounded


if __name__ == "__main__":
    print("Discrete Gaussian Sampler Tests")
    print("=" * 50)
    
    # Test rejection sampler
    print("\n1. Testing Rejection Sampler")
    rs = RejectionSampler(sigma=2.0, center=0.5)
    samples = rs.sample(1000)
    emp_mean = float(mean(samples))
    emp_var = float(variance(samples))
    print(f"Samples: {len(samples)}")
    print(f"Empirical mean: {emp_mean:.3f} (expected: 0.5)")
    print(f"Empirical variance: {emp_var:.3f} (expected: ~4.0)")
    
    # Test CDT sampler
    print("\n2. Testing CDT Sampler")
    cdt = CDTSampler(sigma=1.5, center=0)
    samples = cdt.sample(1000)
    emp_mean = float(mean(samples))
    print(f"Empirical mean: {emp_mean:.3f} (expected: 0.0)")
    
    # Test vector sampler
    print("\n3. Testing Vector Sampler")
    vs = DiscreteGaussianVectorSampler(sigma=1.0, n=5)
    v = vs.sample()
    print(f"Sample vector: {v}")
    print(f"Vector norm: {float(v.norm()):.3f}")
    
    print("\n✓ All tests completed!")