"""
Discrete Gaussian Sampling Utilities Module.

This module provides comprehensive utilities for discrete Gaussian sampling
and analysis on lattices, including:
- 1D discrete Gaussian samplers (exact and approximate)
- Jacobi theta functions and derivatives
- Partition function estimation
- Numerical utilities for stable computation
- Theoretical bounds and analysis tools
- Special samplers for cosets and ellipsoidal distributions

Based on Wang & Ling (2018) and related lattice cryptography literature.
"""

import numpy as np
from scipy.special import erfc
from typing import Union, Tuple, Optional, Callable, List
import warnings


class DiscreteGaussianUtils:
    """Utilities for discrete Gaussian sampling and analysis."""
    
    def __init__(self, precision: float = 1e-10):
        """
        Initialize utilities with numerical precision.
        
        Args:
            precision: Numerical precision for computations
        """
        self.precision = precision
        self.max_iterations = 1000
        
    # 1D Discrete Gaussian Samplers
    
    def sample_discrete_gaussian_1d(self, center: float, sigma: float, 
                                   method: str = 'rejection') -> int:
        """
        Sample from 1D discrete Gaussian distribution.
        
        Args:
            center: Center of the distribution
            sigma: Standard deviation parameter
            method: Sampling method ('rejection', 'cdf', 'alias')
            
        Returns:
            Integer sample from discrete Gaussian
        """
        if method == 'rejection':
            return self._rejection_sample_1d(center, sigma)
        elif method == 'cdf':
            return self._cdf_sample_1d(center, sigma)
        elif method == 'alias':
            return self._alias_sample_1d(center, sigma)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _rejection_sample_1d(self, center: float, sigma: float) -> int:
        """Rejection sampling for 1D discrete Gaussian."""
        # Compute tail bound
        tail_cut = int(np.ceil(center + sigma * np.sqrt(2 * np.log(1/self.precision))))
        
        while True:
            # Sample from continuous Gaussian
            x = np.random.normal(center, sigma)
            x_int = int(np.round(x))
            
            # Acceptance probability
            prob = np.exp(-0.5 * ((x_int - center)**2 - (x - center)**2) / sigma**2)
            
            if np.random.rand() < prob:
                return x_int
    
    def _cdf_sample_1d(self, center: float, sigma: float) -> int:
        """CDF inversion sampling for 1D discrete Gaussian."""
        # Compute support range
        tail_cut = int(np.ceil(sigma * np.sqrt(2 * np.log(1/self.precision))))
        support = range(int(center - tail_cut), int(center + tail_cut + 1))
        
        # Compute probabilities
        probs = np.array([np.exp(-0.5 * (x - center)**2 / sigma**2) for x in support])
        probs /= probs.sum()
        
        # Sample using CDF
        u = np.random.rand()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if u <= cumsum:
                return support[i]
        
        return support[-1]
    
    def _alias_sample_1d(self, center: float, sigma: float) -> int:
        """Alias method sampling for 1D discrete Gaussian."""
        # For repeated sampling from same distribution
        # Implementation of Walker's alias method
        tail_cut = int(np.ceil(sigma * np.sqrt(2 * np.log(1/self.precision))))
        support = list(range(int(center - tail_cut), int(center + tail_cut + 1)))
        n = len(support)
        
        # Compute probabilities
        probs = np.array([np.exp(-0.5 * (x - center)**2 / sigma**2) for x in support])
        probs = probs / probs.sum() * n
        
        # Build alias table
        small = []
        large = []
        for i, p in enumerate(probs):
            if p < 1:
                small.append(i)
            else:
                large.append(i)
        
        alias = np.zeros(n, dtype=int)
        prob = np.ones(n)
        
        while small and large:
            s = small.pop()
            l = large.pop()
            
            prob[s] = probs[s]
            alias[s] = l
            probs[l] = probs[l] + probs[s] - 1
            
            if probs[l] < 1:
                small.append(l)
            else:
                large.append(l)
        
        # Sample
        i = np.random.randint(n)
        if np.random.rand() < prob[i]:
            return support[i]
        else:
            return support[alias[i]]
    
    # Jacobi Theta Functions
    
    def jacobi_theta_3(self, z: complex, tau: complex, 
                      derivatives: int = 0) -> Union[complex, List[complex]]:
        """
        Compute Jacobi theta function ?(z|?) and its derivatives.
        
        ?(z|?) = ?_{n=-}^{} exp(?i?n? + 2?inz)
        
        Args:
            z: Argument
            tau: Lattice parameter (Im(tau) > 0)
            derivatives: Number of derivatives to compute (0, 1, or 2)
            
        Returns:
            Value of theta function or list [?, ?', ?''] if derivatives > 0
        """
        if tau.imag <= 0:
            raise ValueError("Im(tau) must be positive")
        
        # Use Poisson summation for numerical stability
        if tau.imag < 1:
            # Transform to fundamental domain
            return self._theta_transform(z, tau, derivatives)
        
        # Direct summation
        result = [0.0] * (derivatives + 1)
        
        for n in range(-self.max_iterations, self.max_iterations + 1):
            term = np.exp(np.pi * 1j * tau * n**2 + 2 * np.pi * 1j * n * z)
            
            result[0] += term
            if derivatives >= 1:
                result[1] += 2 * np.pi * 1j * n * term
            if derivatives >= 2:
                result[2] += (2 * np.pi * 1j * n)**2 * term
            
            # Check convergence
            if abs(term) < self.precision:
                break
        
        return result if derivatives > 0 else result[0]
    
    def _theta_transform(self, z: complex, tau: complex, derivatives: int) -> Union[complex, List[complex]]:
        """Apply modular transformation for numerical stability."""
        # tau' = -1/tau, z' = z/tau
        tau_prime = -1 / tau
        z_prime = z / tau
        
        # Compute theta function in transformed domain
        theta_vals = self.jacobi_theta_3(z_prime, tau_prime, derivatives)
        
        # Transform back
        prefactor = np.sqrt(-1j / tau) * np.exp(np.pi * 1j * z**2 / tau)
        
        if derivatives == 0:
            return prefactor * theta_vals
        else:
            # Handle derivative transformations
            result = [prefactor * theta_vals[0]]
            if derivatives >= 1:
                result.append(prefactor * (theta_vals[1] / tau + 2 * np.pi * 1j * z * theta_vals[0] / tau))
            if derivatives >= 2:
                # Second derivative transformation (more complex)
                result.append(prefactor * (theta_vals[2] / tau**2 + 
                                         4 * np.pi * 1j * z * theta_vals[1] / tau**2 +
                                         (2 * np.pi * 1j / tau + (2 * np.pi * 1j * z / tau)**2) * theta_vals[0]))
            return result
    
    def riemann_theta(self, z: np.ndarray, omega: np.ndarray) -> complex:
        """
        Compute Riemann theta function for general lattices.
        
        ?(z|?) = ?_{nZ^g} exp(?i n^T ? n + 2?i n^T z)
        
        Args:
            z: Argument vector (g-dimensional)
            omega: Period matrix (g?g, symmetric with positive imaginary part)
            
        Returns:
            Value of Riemann theta function
        """
        g = len(z)
        if omega.shape != (g, g):
            raise ValueError("Dimension mismatch between z and omega")
        
        # Check that Im(Omega) is positive definite
        omega_imag = omega.imag
        if not np.all(np.linalg.eigvals(omega_imag) > 0):
            raise ValueError("Im(Omega) must be positive definite")
        
        # Compute using lattice reduction for efficiency
        # This is a simplified version - full implementation would use
        # Siegel reduction and more sophisticated techniques
        
        result = 0.0
        bound = int(np.ceil(np.sqrt(-2 * np.log(self.precision) / np.pi / omega_imag.min())))
        
        for indices in np.ndindex(*([2*bound+1]*g)):
            n = np.array(indices) - bound
            exponent = np.pi * 1j * (n.T @ omega @ n) + 2 * np.pi * 1j * (n.T @ z)
            result += np.exp(exponent)
        
        return result
    
    # Partition Function Estimation
    
    def partition_function(self, lattice_basis: np.ndarray, sigma: float,
                          method: str = 'theta') -> float:
        """
        Estimate partition function Z_?(?) = ?_{v?} exp(-?||v||?/?).
        
        Args:
            lattice_basis: Basis matrix of the lattice
            sigma: Gaussian parameter
            method: Estimation method ('theta', 'monte_carlo', 'bounds')
            
        Returns:
            Estimate of partition function
        """
        if method == 'theta':
            return self._partition_theta(lattice_basis, sigma)
        elif method == 'monte_carlo':
            return self._partition_monte_carlo(lattice_basis, sigma)
        elif method == 'bounds':
            return self._partition_bounds(lattice_basis, sigma)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _partition_theta(self, basis: np.ndarray, sigma: float) -> float:
        """Compute partition function using theta functions."""
        n = basis.shape[0]
        gram = basis @ basis.T
        
        # For general lattices, this requires Riemann theta
        # Here we implement for diagonal case
        if not np.allclose(gram, np.diag(np.diag(gram))):
            warnings.warn("Theta method currently supports only orthogonal lattices")
            return self._partition_monte_carlo(basis, sigma)
        
        # Product of 1D theta functions
        result = 1.0
        for i in range(n):
            norm_i = np.sqrt(gram[i, i])
            tau_i = 1j * sigma**2 / (2 * norm_i**2)
            result *= self.jacobi_theta_3(0, tau_i)
        
        return result
    
    def _partition_monte_carlo(self, basis: np.ndarray, sigma: float) -> float:
        """Monte Carlo estimation of partition function."""
        n_samples = 10000
        
        # Importance sampling from continuous Gaussian
        samples = np.random.normal(0, sigma, (n_samples, basis.shape[0]))
        
        # Decode to nearest lattice points
        lattice_points = []
        for sample in samples:
            # Simple rounding decoder (should use proper CVP solver)
            coeffs = np.linalg.solve(basis.T, sample)
            coeffs_int = np.round(coeffs).astype(int)
            lattice_point = basis.T @ coeffs_int
            lattice_points.append(lattice_point)
        
        # Estimate partition function
        weights = []
        for i, point in enumerate(lattice_points):
            weight = np.exp(-np.pi * np.sum(point**2) / sigma**2 + 
                           np.pi * np.sum(samples[i]**2) / sigma**2)
            weights.append(weight)
        
        return np.mean(weights) * (sigma * np.sqrt(2 * np.pi))**basis.shape[0]
    
    def _partition_bounds(self, basis: np.ndarray, sigma: float) -> Tuple[float, float]:
        """Compute bounds on partition function."""
        n = basis.shape[0]
        det = np.linalg.det(basis)
        
        # Lower bound: single point contribution
        lower = 1.0
        
        # Upper bound: Gaussian heuristic
        upper = (sigma * np.sqrt(2 * np.pi))**n / abs(det)
        
        return lower, upper
    
    # Numerical Utilities
    
    def log_sum_exp(self, log_values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Numerically stable computation of log(sum(exp(values))).
        
        Args:
            log_values: Array of log values
            axis: Axis along which to sum
            
        Returns:
            log(sum(exp(log_values)))
        """
        max_val = np.max(log_values, axis=axis, keepdims=True)
        return np.squeeze(max_val) + np.log(np.sum(np.exp(log_values - max_val), axis=axis))
    
    def stable_softmax(self, logits: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Numerically stable softmax computation.
        
        Args:
            logits: Input logits
            axis: Axis along which to compute softmax
            
        Returns:
            Softmax probabilities
        """
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
    
    def stable_log_probability(self, x: np.ndarray, center: np.ndarray, 
                              sigma: float) -> float:
        """
        Compute log probability for discrete Gaussian in numerically stable way.
        
        Args:
            x: Point to evaluate
            center: Center of distribution
            sigma: Standard deviation parameter
            
        Returns:
            Log probability (unnormalized)
        """
        return -0.5 * np.sum((x - center)**2) / sigma**2
    
    # Theoretical Bounds
    
    def smoothing_parameter(self, basis: np.ndarray, epsilon: float = 0.01) -> float:
        """
        Compute smoothing parameter ?_?(?) for lattice.
        
        Args:
            basis: Lattice basis
            epsilon: Smoothing parameter (typically 2^-40 to 2^-80)
            
        Returns:
            Smoothing parameter ?_?(?)
        """
        # Compute using dual lattice minimum
        gram = basis @ basis.T
        gram_inv = np.linalg.inv(gram)
        dual_basis = basis @ gram_inv
        
        # Estimate using covering radius of dual
        # This is a heuristic - exact computation is #P-hard
        n = basis.shape[0]
        dual_det = 1 / np.sqrt(np.linalg.det(gram))
        
        # Gaussian heuristic for dual minimum
        lambda_1_dual = np.sqrt(n / (2 * np.pi * np.e)) * dual_det**(1/n)
        
        # Smoothing parameter bound
        eta = np.sqrt(np.log(2 * n * (1 + 1/epsilon)) / np.pi) / lambda_1_dual
        
        return eta
    
    def statistical_distance(self, samples1: np.ndarray, samples2: np.ndarray,
                           lattice_basis: np.ndarray) -> float:
        """
        Estimate statistical distance between two discrete distributions on lattice.
        
        Args:
            samples1: Samples from first distribution
            samples2: Samples from second distribution
            lattice_basis: Basis of the lattice
            
        Returns:
            Estimate of total variation distance
        """
        # Decode samples to lattice points
        points1 = self._decode_to_lattice(samples1, lattice_basis)
        points2 = self._decode_to_lattice(samples2, lattice_basis)
        
        # Estimate probabilities
        unique1, counts1 = np.unique(points1, axis=0, return_counts=True)
        unique2, counts2 = np.unique(points2, axis=0, return_counts=True)
        
        prob1 = {tuple(p): c / len(points1) for p, c in zip(unique1, counts1)}
        prob2 = {tuple(p): c / len(points2) for p, c in zip(unique2, counts2)}
        
        # Compute TVD
        all_points = set(prob1.keys()) | set(prob2.keys())
        tvd = 0.5 * sum(abs(prob1.get(p, 0) - prob2.get(p, 0)) for p in all_points)
        
        return tvd
    
    def _decode_to_lattice(self, points: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Decode continuous points to nearest lattice points."""
        decoded = []
        for point in points:
            coeffs = np.linalg.solve(basis.T, point)
            coeffs_int = np.round(coeffs).astype(int)
            lattice_point = basis.T @ coeffs_int
            decoded.append(lattice_point)
        return np.array(decoded)
    
    def rho_inverse(self, lattice_basis: np.ndarray, sigma: float, t: float) -> float:
        """
        Compute ?_?^(-1)(?, t) = min{r : ?_?(? ) B(0,r)) e t}.
        
        Args:
            lattice_basis: Basis of the lattice
            sigma: Gaussian parameter
            t: Threshold value (0 < t < 1)
            
        Returns:
            Radius r such that mass in ball B(0,r) is at least t
        """
        if not 0 < t < 1:
            raise ValueError("Threshold t must be in (0, 1)")
        
        # Binary search for radius
        det = np.abs(np.linalg.det(lattice_basis))
        n = lattice_basis.shape[0]
        
        # Initial bounds
        r_min = 0
        r_max = sigma * np.sqrt(2 * n * np.log(1 / (1 - t)))
        
        while r_max - r_min > self.precision:
            r_mid = (r_min + r_max) / 2
            
            # Estimate mass in ball
            mass = self._estimate_mass_in_ball(lattice_basis, sigma, r_mid)
            
            if mass < t:
                r_min = r_mid
            else:
                r_max = r_mid
        
        return r_max
    
    def _estimate_mass_in_ball(self, basis: np.ndarray, sigma: float, radius: float) -> float:
        """Estimate Gaussian mass within ball of given radius."""
        # Use importance sampling
        n_samples = 1000
        count = 0
        
        for _ in range(n_samples):
            # Sample from discrete Gaussian
            sample = self.sample_discrete_gaussian_lattice(basis, sigma)
            if np.linalg.norm(sample) <= radius:
                count += 1
        
        return count / n_samples
    
    # Special Samplers
    
    def sample_discrete_gaussian_coset(self, lattice_basis: np.ndarray,
                                     coset_shift: np.ndarray,
                                     sigma: float) -> np.ndarray:
        """
        Sample from discrete Gaussian on coset ? + c.
        
        Args:
            lattice_basis: Basis of the lattice ?
            coset_shift: Coset shift vector c
            sigma: Gaussian parameter
            
        Returns:
            Sample from D_{?+c,?}
        """
        # Sample from D_{?,?,c} using rejection
        max_attempts = 1000
        
        for _ in range(max_attempts):
            # Sample from continuous Gaussian centered at c
            x = np.random.normal(coset_shift, sigma)
            
            # Decode to nearest lattice point in ?
            coeffs = np.linalg.solve(lattice_basis.T, x - coset_shift)
            coeffs_int = np.round(coeffs).astype(int)
            lattice_point = lattice_basis.T @ coeffs_int
            
            # Coset point
            y = lattice_point + coset_shift
            
            # Accept with appropriate probability
            prob = np.exp(-0.5 * (np.sum((y - x)**2) - np.sum((coset_shift - x)**2)) / sigma**2)
            
            if np.random.rand() < prob:
                return y
        
        raise RuntimeError("Failed to sample from coset Gaussian")
    
    def sample_ellipsoidal_gaussian(self, lattice_basis: np.ndarray,
                                  covariance: np.ndarray) -> np.ndarray:
        """
        Sample from ellipsoidal discrete Gaussian on lattice.
        
        Args:
            lattice_basis: Basis of the lattice
            covariance: Covariance matrix ?
            
        Returns:
            Sample from ellipsoidal discrete Gaussian
        """
        n = lattice_basis.shape[0]
        
        # Decompose ? = AA^T
        A = np.linalg.cholesky(covariance)
        
        # Transform to spherical case
        transformed_basis = np.linalg.solve(A, lattice_basis.T).T
        
        # Sample from spherical Gaussian on transformed lattice
        z = self.sample_discrete_gaussian_lattice(transformed_basis, 1.0)
        
        # Transform back
        return A @ z
    
    def sample_discrete_gaussian_lattice(self, basis: np.ndarray, sigma: float) -> np.ndarray:
        """
        Generic lattice Gaussian sampler (uses Klein's algorithm).
        
        Args:
            basis: Lattice basis
            sigma: Gaussian parameter
            
        Returns:
            Sample from discrete Gaussian on lattice
        """
        # This is a simplified version - full implementation would use Klein's algorithm
        n = basis.shape[0]
        
        # Sample continuous Gaussian
        x = np.random.normal(0, sigma, n)
        
        # Decode to nearest lattice point (Babai's nearest plane)
        coeffs = np.linalg.solve(basis.T, x)
        coeffs_int = np.round(coeffs).astype(int)
        
        return basis.T @ coeffs_int
    
    # Analytical Tools
    
    def gaussian_mass_above(self, threshold: float, sigma: float, 
                           dimension: int) -> float:
        """
        Compute Pr[||X|| > t] for continuous Gaussian in R^n.
        
        Args:
            threshold: Threshold value t
            sigma: Standard deviation
            dimension: Dimension n
            
        Returns:
            Probability mass above threshold
        """
        # Chi distribution CDF
        from scipy.stats import chi
        return 1 - chi.cdf(threshold / sigma, dimension)
    
    def discrete_gaussian_moments(self, lattice_basis: np.ndarray, sigma: float,
                                order: int = 2) -> np.ndarray:
        """
        Compute moments of discrete Gaussian distribution.
        
        Args:
            lattice_basis: Lattice basis
            sigma: Gaussian parameter
            order: Moment order (1=mean, 2=second moment, etc.)
            
        Returns:
            Array of moments up to specified order
        """
        if order > 4:
            warnings.warn("High-order moments may be inaccurate")
        
        # For lattice Gaussian, odd moments are zero by symmetry
        # Even moments can be computed using theta function derivatives
        
        moments = []
        for k in range(1, order + 1):
            if k % 2 == 1:
                moments.append(np.zeros(lattice_basis.shape[0]))
            else:
                # Even moment - requires theta function derivatives
                # This is a placeholder - full implementation would use
                # derivatives of partition function
                moment = (sigma**k) * np.ones(lattice_basis.shape[0])
                moments.append(moment)
        
        return moments
    
    def mixing_time_bound(self, basis: np.ndarray, sigma: float, 
                         epsilon: float = 0.01) -> int:
        """
        Theoretical upper bound on mixing time for IMHK.
        
        Args:
            basis: Lattice basis
            sigma: Gaussian parameter
            epsilon: Target TV distance
            
        Returns:
            Upper bound on mixing time
        """
        n = basis.shape[0]
        
        # Compute smoothing parameter
        eta = self.smoothing_parameter(basis, epsilon / 4)
        
        if sigma < eta:
            warnings.warn(f"? = {sigma} < ?_{epsilon/4}(?) = {eta}, "
                         f"mixing time bound may be loose")
        
        # Compute spectral gap bound
        gap_bound = 1 - 2 * epsilon if sigma >= eta else epsilon**2
        
        # Mixing time bound
        t_mix = int(np.ceil(np.log(1 / epsilon) / gap_bound))
        
        return t_mix


# Testing and Validation

def test_discrete_gaussian_utils():
    """Test suite for discrete Gaussian utilities."""
    utils = DiscreteGaussianUtils()
    
    print("Testing Discrete Gaussian Utilities...")
    
    # Test 1D samplers
    print("\n1. Testing 1D discrete Gaussian samplers:")
    center = 2.5
    sigma = 3.0
    n_samples = 10000
    
    for method in ['rejection', 'cdf', 'alias']:
        samples = [utils.sample_discrete_gaussian_1d(center, sigma, method) 
                  for _ in range(n_samples)]
        mean = np.mean(samples)
        var = np.var(samples)
        print(f"  {method}: mean={mean:.3f} (expectedH{center:.3f}), "
              f"var={var:.3f} (expectedH{sigma**2:.3f})")
    
    # Test Jacobi theta function
    print("\n2. Testing Jacobi theta function:")
    z = 0.1 + 0.2j
    tau = 0.5j
    theta_vals = utils.jacobi_theta_3(z, tau, derivatives=2)
    print(f"  ?({z}|{tau}) = {theta_vals[0]:.6f}")
    print(f"  ?'({z}|{tau}) = {theta_vals[1]:.6f}")
    print(f"  ?''({z}|{tau}) = {theta_vals[2]:.6f}")
    
    # Test partition function
    print("\n3. Testing partition function estimation:")
    basis = np.diag([1, 2, 3])  # Simple orthogonal lattice
    sigma = 2.0
    
    Z_theta = utils.partition_function(basis, sigma, method='theta')
    Z_mc = utils.partition_function(basis, sigma, method='monte_carlo')
    Z_bounds = utils.partition_function(basis, sigma, method='bounds')
    
    print(f"  Theta method: Z = {Z_theta:.6f}")
    print(f"  Monte Carlo: Z = {Z_mc:.6f}")
    print(f"  Bounds: Z  [{Z_bounds[0]:.6f}, {Z_bounds[1]:.6f}]")
    
    # Test numerical utilities
    print("\n4. Testing numerical utilities:")
    log_values = np.array([-1000, -999, -998])
    lse = utils.log_sum_exp(log_values)
    print(f"  log_sum_exp({log_values}) = {lse:.6f}")
    
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    probs = utils.stable_softmax(logits)
    print(f"  softmax({logits}) = {probs}")
    
    # Test smoothing parameter
    print("\n5. Testing smoothing parameter:")
    eta = utils.smoothing_parameter(basis, epsilon=0.01)
    print(f"  ?_0.01(?) = {eta:.6f}")
    
    # Test coset sampling
    print("\n6. Testing coset Gaussian sampling:")
    coset_shift = np.array([0.5, 0.5, 0.5])
    sample = utils.sample_discrete_gaussian_coset(basis, coset_shift, sigma)
    print(f"  Sample from ? + c: {sample}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_discrete_gaussian_utils()