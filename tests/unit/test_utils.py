"""
Unit tests for utility modules.

Tests cover discrete Gaussian utilities, helper functions, and
numerical utilities used throughout the project.
"""

import numpy as np
import pytest
from typing import List, Tuple, Optional
from unittest.mock import Mock, patch
from scipy import stats, special

from samplers.utils import DiscreteGaussianUtils


class TestDiscreteGaussianUtils:
    """Test discrete Gaussian utility functions."""
    
    def test_discrete_gaussian_utils_initialization(self):
        """Test DiscreteGaussianUtils initialization."""
        utils = DiscreteGaussianUtils()
        
        # Check that all expected methods exist
        expected_methods = [
            'sample_discrete_gaussian_1d',
            'jacobi_theta3',
            'compute_partition_function',
            'discrete_gaussian_tail_bound',
            'sample_integer_lattice'
        ]
        
        for method in expected_methods:
            assert hasattr(utils, method), f"Missing method: {method}"
    
    @pytest.mark.statistical
    def test_discrete_gaussian_1d_sampling(self, statistical_config, tolerance_config):
        """Test 1D discrete Gaussian sampling correctness."""
        utils = DiscreteGaussianUtils()
        sigma = 2.0
        n_samples = statistical_config['n_samples']
        
        # Generate samples
        samples = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(n_samples)]
        
        # Test that all samples are integers
        assert all(isinstance(s, (int, np.integer)) for s in samples), \
            "All samples must be integers"
        
        # Test sample mean (should be close to 0)
        sample_mean = np.mean(samples)
        std_error = sigma / np.sqrt(n_samples)
        
        assert abs(sample_mean) < 3 * std_error, \
            f"Sample mean {sample_mean} deviates too much from 0"
        
        # Test sample variance (should be close to sigma^2)
        sample_var = np.var(samples)
        expected_var = sigma**2
        
        # Allow for statistical fluctuation
        var_tolerance = statistical_config['statistical_rtol']
        assert abs(sample_var - expected_var) < var_tolerance * expected_var, \
            f"Sample variance {sample_var} deviates too much from expected {expected_var}"
    
    def test_discrete_gaussian_1d_different_sigmas(self):
        """Test 1D discrete Gaussian with different sigma values."""
        utils = DiscreteGaussianUtils()
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for sigma in sigmas:
            # Generate fewer samples for this test
            samples = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(1000)]
            
            # Check that samples are integers
            assert all(isinstance(s, (int, np.integer)) for s in samples)
            
            # Check that variance scales appropriately
            sample_std = np.std(samples)
            
            # Should be roughly proportional to sigma
            # (allowing for significant tolerance due to discrete nature)
            assert 0.5 * sigma < sample_std < 3.0 * sigma, \
                f"Sample std {sample_std} not reasonable for sigma={sigma}"
    
    @pytest.mark.edge_case
    def test_discrete_gaussian_1d_edge_cases(self):
        """Test 1D discrete Gaussian edge cases."""
        utils = DiscreteGaussianUtils()
        
        # Very small sigma
        small_sigma = 1e-3
        small_samples = [utils.sample_discrete_gaussian_1d(small_sigma) for _ in range(100)]
        
        # Most samples should be 0 for very small sigma
        zero_count = sum(1 for s in small_samples if s == 0)
        assert zero_count > 80, "Very small sigma should concentrate samples at 0"
        
        # Very large sigma
        large_sigma = 100.0
        large_sample = utils.sample_discrete_gaussian_1d(large_sigma)
        assert isinstance(large_sample, (int, np.integer)), \
            "Large sigma should still produce integer samples"
        
        # Invalid sigma values
        with pytest.raises(ValueError, match="sigma must be positive"):
            utils.sample_discrete_gaussian_1d(0.0)
            
        with pytest.raises(ValueError, match="sigma must be positive"):
            utils.sample_discrete_gaussian_1d(-1.0)
    
    @pytest.mark.numerical
    def test_jacobi_theta3_function(self, tolerance_config):
        """Test Jacobi theta3 function computation."""
        utils = DiscreteGaussianUtils()
        
        # Test with known values and properties
        test_cases = [
            (0.0, 1.0),   # theta3(0, 1)
            (0.0, 2.0),   # theta3(0, 2)
            (1.0, 1.0),   # theta3(1, 1)
            (0.5, 0.5),   # theta3(0.5, 0.5)
        ]
        
        for z, tau in test_cases:
            theta = utils.jacobi_theta3(z, tau)
            
            # Basic properties
            assert np.isfinite(theta), f"Theta function not finite for z={z}, tau={tau}"
            assert theta.real > 0, f"Theta function not positive for z={z}, tau={tau}"
            
            # Theta function should be real for real inputs
            if np.isreal(z) and np.isreal(tau) and tau > 0:
                assert np.isreal(theta), f"Theta function should be real for real inputs"
    
    def test_jacobi_theta3_properties(self, tolerance_config):
        """Test mathematical properties of Jacobi theta3 function."""
        utils = DiscreteGaussianUtils()
        
        # Test periodicity in z: theta3(z + 2π, tau) = theta3(z, tau)
        z, tau = 0.5, 1.0
        theta1 = utils.jacobi_theta3(z, tau)
        theta2 = utils.jacobi_theta3(z + 2*np.pi, tau)
        
        np.testing.assert_allclose(
            theta1, theta2, rtol=tolerance_config['rtol'],
            err_msg="Jacobi theta3 should be periodic in z"
        )
        
        # Test symmetry: theta3(-z, tau) = theta3(z, tau)
        theta_pos = utils.jacobi_theta3(1.0, 1.0)
        theta_neg = utils.jacobi_theta3(-1.0, 1.0)
        
        np.testing.assert_allclose(
            theta_pos, theta_neg, rtol=tolerance_config['rtol'],
            err_msg="Jacobi theta3 should be even in z"
        )
    
    @pytest.mark.edge_case
    def test_jacobi_theta3_edge_cases(self):
        """Test Jacobi theta3 function edge cases."""
        utils = DiscreteGaussianUtils()
        
        # Very small tau (should handle gracefully)
        theta_small_tau = utils.jacobi_theta3(0.0, 1e-6)
        assert np.isfinite(theta_small_tau), "Should handle small tau"
        
        # Large tau
        theta_large_tau = utils.jacobi_theta3(0.0, 100.0)
        assert np.isfinite(theta_large_tau), "Should handle large tau"
        
        # Invalid tau (should raise error)
        with pytest.raises((ValueError, ZeroDivisionError)):
            utils.jacobi_theta3(0.0, 0.0)
            
        with pytest.raises((ValueError, ZeroDivisionError)):
            utils.jacobi_theta3(0.0, -1.0)
    
    def test_partition_function_computation(self, tolerance_config):
        """Test partition function computation."""
        from lattices.identity import IdentityLattice
        
        utils = DiscreteGaussianUtils()
        
        # Test on identity lattice (known case)
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        
        Z = utils.compute_partition_function(lattice, sigma)
        
        # Basic properties
        assert Z > 0, "Partition function must be positive"
        assert np.isfinite(Z), "Partition function must be finite"
        
        # For identity lattice, can compare with theoretical approximation
        # Z ≈ (σ√(2π))^n for large σ
        n = lattice.get_dimension()
        theoretical_approx = (sigma * np.sqrt(2 * np.pi))**n
        
        # Should be in reasonable range (discrete vs continuous approximation)
        assert 0.1 * theoretical_approx < Z < 10 * theoretical_approx, \
            f"Partition function {Z} not in reasonable range compared to {theoretical_approx}"
    
    def test_partition_function_different_sigmas(self):
        """Test partition function with different sigma values."""
        from lattices.identity import IdentityLattice
        
        utils = DiscreteGaussianUtils()
        lattice = IdentityLattice(dimension=2)
        
        sigmas = [0.5, 1.0, 2.0, 5.0]
        partition_functions = []
        
        for sigma in sigmas:
            Z = utils.compute_partition_function(lattice, sigma)
            partition_functions.append(Z)
            
            assert Z > 0, f"Partition function must be positive for sigma={sigma}"
            assert np.isfinite(Z), f"Partition function must be finite for sigma={sigma}"
        
        # Larger sigma should generally give larger partition function
        for i in range(len(partition_functions) - 1):
            # Allow some tolerance due to numerical computation
            assert partition_functions[i+1] >= partition_functions[i] * 0.8, \
                "Partition function should generally increase with sigma"
    
    def test_discrete_gaussian_tail_bound(self, tolerance_config):
        """Test discrete Gaussian tail bound computation."""
        utils = DiscreteGaussianUtils()
        sigma = 1.0
        
        # Test tail bounds for different radii
        radii = [1, 2, 3, 5, 10]
        previous_bound = 1.0
        
        for r in radii:
            bound = utils.discrete_gaussian_tail_bound(r, sigma)
            
            # Basic properties
            assert 0 <= bound <= 1, f"Tail bound {bound} not in [0,1] for r={r}"
            assert np.isfinite(bound), f"Tail bound must be finite for r={r}"
            
            # Should be decreasing with radius
            assert bound <= previous_bound + tolerance_config['rtol'], \
                f"Tail bound not decreasing: {bound} > {previous_bound} for r={r}"
            previous_bound = bound
        
        # Very large radius should give very small bound
        large_bound = utils.discrete_gaussian_tail_bound(100, sigma)
        assert large_bound < 1e-10, "Tail bound should be very small for large radius"
    
    def test_discrete_gaussian_tail_bound_different_sigmas(self):
        """Test tail bounds with different sigma values."""
        utils = DiscreteGaussianUtils()
        radius = 5
        
        sigmas = [0.5, 1.0, 2.0, 5.0]
        
        for sigma in sigmas:
            bound = utils.discrete_gaussian_tail_bound(radius, sigma)
            
            assert 0 <= bound <= 1, f"Invalid tail bound for sigma={sigma}"
            assert np.isfinite(bound), f"Tail bound not finite for sigma={sigma}"
    
    @pytest.mark.edge_case
    def test_discrete_gaussian_tail_bound_edge_cases(self):
        """Test discrete Gaussian tail bound edge cases."""
        utils = DiscreteGaussianUtils()
        
        # Zero radius
        bound_zero = utils.discrete_gaussian_tail_bound(0, 1.0)
        assert bound_zero == 1.0, "Tail bound should be 1 for radius 0"
        
        # Very small sigma
        bound_small_sigma = utils.discrete_gaussian_tail_bound(1, 1e-6)
        assert 0 <= bound_small_sigma <= 1, "Tail bound should be valid for small sigma"
        
        # Very large sigma
        bound_large_sigma = utils.discrete_gaussian_tail_bound(1, 100.0)
        assert 0 <= bound_large_sigma <= 1, "Tail bound should be valid for large sigma"
        
        # Invalid inputs
        with pytest.raises(ValueError):
            utils.discrete_gaussian_tail_bound(-1, 1.0)  # Negative radius
            
        with pytest.raises(ValueError):
            utils.discrete_gaussian_tail_bound(1, 0.0)   # Zero sigma
    
    def test_sample_integer_lattice(self, statistical_config):
        """Test sampling from integer lattice Z^n."""
        utils = DiscreteGaussianUtils()
        
        dimensions = [1, 2, 3, 5]
        sigma = 1.0
        
        for dim in dimensions:
            # Generate samples
            samples = [utils.sample_integer_lattice(dim, sigma) for _ in range(1000)]
            
            # Check dimensions
            for sample in samples:
                assert len(sample) == dim, f"Sample has wrong dimension for dim={dim}"
                assert all(isinstance(x, (int, np.integer)) for x in sample), \
                    "All coordinates must be integers"
            
            # Check statistical properties (for reasonable dimensions)
            if dim <= 3:
                samples_array = np.array(samples)
                
                # Check mean (should be close to zero)
                sample_mean = np.mean(samples_array, axis=0)
                assert np.allclose(sample_mean, 0, atol=0.2), \
                    f"Sample mean deviates too much for dim={dim}"
                
                # Check that samples are spread out
                sample_std = np.std(samples_array, axis=0)
                assert np.all(sample_std > 0.5 * sigma), \
                    f"Samples not spread enough for dim={dim}"
    
    @pytest.mark.edge_case
    def test_sample_integer_lattice_edge_cases(self):
        """Test integer lattice sampling edge cases."""
        utils = DiscreteGaussianUtils()
        
        # Very small dimension
        sample_1d = utils.sample_integer_lattice(1, 1.0)
        assert len(sample_1d) == 1
        assert isinstance(sample_1d[0], (int, np.integer))
        
        # Large dimension (should still work)
        sample_large = utils.sample_integer_lattice(10, 1.0)
        assert len(sample_large) == 10
        assert all(isinstance(x, (int, np.integer)) for x in sample_large)
        
        # Very small sigma
        sample_small_sigma = utils.sample_integer_lattice(2, 1e-6)
        # Should concentrate at origin
        assert all(abs(x) <= 1 for x in sample_small_sigma), \
            "Very small sigma should give samples near origin"
        
        # Invalid inputs
        with pytest.raises(ValueError):
            utils.sample_integer_lattice(0, 1.0)  # Zero dimension
            
        with pytest.raises(ValueError):
            utils.sample_integer_lattice(-1, 1.0)  # Negative dimension
            
        with pytest.raises(ValueError):
            utils.sample_integer_lattice(2, 0.0)  # Zero sigma
    
    def test_special_lattice_samplers(self):
        """Test special case samplers if implemented."""
        utils = DiscreteGaussianUtils()
        
        # Test if special samplers are available
        if hasattr(utils, 'sample_root_lattice'):
            # Test A2 root lattice sampler
            sample_a2 = utils.sample_root_lattice('A2', 1.0)
            assert len(sample_a2) >= 2, "A2 lattice should have dimension >= 2"
            
        if hasattr(utils, 'sample_hexagonal_lattice'):
            # Test hexagonal lattice sampler
            sample_hex = utils.sample_hexagonal_lattice(1.0)
            assert len(sample_hex) == 2, "Hexagonal lattice should be 2D"
        
        if hasattr(utils, 'sample_bcc_lattice'):
            # Test body-centered cubic lattice sampler
            sample_bcc = utils.sample_bcc_lattice(1.0)
            assert len(sample_bcc) == 3, "BCC lattice should be 3D"


class TestNumericalUtilities:
    """Test numerical utility functions."""
    
    def test_log_sum_exp_stability(self):
        """Test numerically stable log-sum-exp computation."""
        utils = DiscreteGaussianUtils()
        
        # Test with large values (would overflow without log-sum-exp)
        large_values = np.array([1000, 1001, 999])
        
        if hasattr(utils, 'log_sum_exp'):
            result = utils.log_sum_exp(large_values)
            
            # Should be finite
            assert np.isfinite(result), "log-sum-exp should handle large values"
            
            # Should be approximately max + log(n) for similar large values
            expected_approx = np.max(large_values) + np.log(len(large_values))
            assert abs(result - expected_approx) < 10, \
                "log-sum-exp result not in expected range"
    
    def test_gamma_function_computation(self):
        """Test gamma function computation if available."""
        utils = DiscreteGaussianUtils()
        
        if hasattr(utils, 'log_gamma'):
            # Test with positive values
            test_values = [0.5, 1.0, 2.0, 5.5, 10.0]
            
            for x in test_values:
                log_gamma_x = utils.log_gamma(x)
                
                assert np.isfinite(log_gamma_x), f"log_gamma({x}) should be finite"
                
                # Compare with scipy if available
                expected = special.loggamma(x)
                np.testing.assert_allclose(
                    log_gamma_x, expected, rtol=1e-10,
                    err_msg=f"log_gamma({x}) differs from scipy"
                )
    
    def test_bessel_function_computation(self):
        """Test Bessel function computation if available."""
        utils = DiscreteGaussianUtils()
        
        if hasattr(utils, 'modified_bessel_i0'):
            # Test modified Bessel function I_0
            test_values = [0.0, 0.5, 1.0, 2.0, 5.0]
            
            for x in test_values:
                i0_x = utils.modified_bessel_i0(x)
                
                assert np.isfinite(i0_x), f"I_0({x}) should be finite"
                assert i0_x > 0, f"I_0({x}) should be positive"
                
                # I_0(0) = 1
                if x == 0.0:
                    np.testing.assert_allclose(i0_x, 1.0, rtol=1e-10)


@pytest.mark.reproducibility
class TestUtilsReproducibility:
    """Test reproducibility of utility functions."""
    
    def test_discrete_gaussian_sampling_reproducibility(self, test_seed):
        """Test that discrete Gaussian sampling is reproducible."""
        utils = DiscreteGaussianUtils()
        sigma = 1.0
        
        # Generate samples with same seed
        np.random.seed(test_seed)
        samples1 = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(100)]
        
        np.random.seed(test_seed)
        samples2 = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(100)]
        
        # Samples should be identical
        assert samples1 == samples2, "Discrete Gaussian sampling should be reproducible"
    
    def test_partition_function_reproducibility(self, test_seed):
        """Test that partition function computation is reproducible."""
        from lattices.identity import IdentityLattice
        
        utils = DiscreteGaussianUtils()
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        
        # Compute partition function twice
        np.random.seed(test_seed)
        Z1 = utils.compute_partition_function(lattice, sigma)
        
        np.random.seed(test_seed)
        Z2 = utils.compute_partition_function(lattice, sigma)
        
        # Results should be identical
        np.testing.assert_allclose(Z1, Z2, rtol=1e-15, 
                                 err_msg="Partition function computation should be reproducible")


@pytest.mark.performance
class TestUtilsPerformance:
    """Test performance of utility functions."""
    
    def test_discrete_gaussian_sampling_performance(self, performance_config):
        """Test performance of discrete Gaussian sampling."""
        import time
        
        utils = DiscreteGaussianUtils()
        sigma = 1.0
        n_samples = 10000
        
        # Time the sampling
        start_time = time.time()
        samples = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(n_samples)]
        elapsed_time = time.time() - start_time
        
        # Should be reasonably fast
        max_time = performance_config['max_time_simple']
        assert elapsed_time < max_time, \
            f"Discrete Gaussian sampling too slow: {elapsed_time:.2f}s for {n_samples} samples"
        
        # Check that we actually got samples
        assert len(samples) == n_samples
        assert all(isinstance(s, (int, np.integer)) for s in samples)
    
    def test_jacobi_theta_performance(self, performance_config):
        """Test performance of Jacobi theta function computation."""
        import time
        
        utils = DiscreteGaussianUtils()
        
        # Test with multiple evaluations
        n_evaluations = 1000
        test_points = [(np.random.rand(), 0.5 + np.random.rand()) for _ in range(n_evaluations)]
        
        start_time = time.time()
        for z, tau in test_points:
            theta = utils.jacobi_theta3(z, tau)
        elapsed_time = time.time() - start_time
        
        # Should be reasonably fast
        max_time = performance_config['max_time_simple']
        assert elapsed_time < max_time, \
            f"Jacobi theta computation too slow: {elapsed_time:.2f}s for {n_evaluations} evaluations"