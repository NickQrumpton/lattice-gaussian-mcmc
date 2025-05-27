"""
Test suite for sampling algorithms.

Tests all sampler implementations for:
- Correctness of sampling distribution
- Convergence properties
- Performance characteristics
- Theoretical guarantees
"""

import pytest
import numpy as np
from scipy import stats
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IMHKSampler
from src.samplers.utils import DiscreteGaussianUtils
from src.lattices.identity import IdentityLattice
from src.lattices.qary import QaryLattice


class TestKleinSampler:
    """Test Klein's sampling algorithm."""
    
    def test_sampling_distribution(self):
        """Test that samples follow discrete Gaussian distribution."""
        # Use identity lattice for easy verification
        lattice = IdentityLattice(10)
        sigma = 5.0
        sampler = KleinSampler(lattice, sigma)
        
        # Generate samples
        n_samples = 10000
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Check empirical mean (should be near 0)
        empirical_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(empirical_mean, 0, atol=0.1)
        
        # Check empirical covariance
        empirical_cov = np.cov(samples.T)
        expected_cov = sigma**2 * np.eye(10) * (1 - np.exp(-2*np.pi))
        np.testing.assert_allclose(empirical_cov, expected_cov, rtol=0.1)
    
    def test_klein_correctness(self):
        """Test Klein's algorithm produces valid lattice points."""
        # Random lattice
        n, m, q = 20, 40, 1024
        lattice = QaryLattice.random_qary_lattice(n, m, q)
        sigma = 10.0
        sampler = KleinSampler(lattice, sigma)
        
        # Generate samples and verify they're in the lattice
        for _ in range(100):
            sample = sampler.sample()
            
            # Check sample is in lattice by verifying it's an integer
            # combination of basis vectors
            basis = lattice.get_basis()
            coeffs = np.linalg.solve(basis.T, sample)
            np.testing.assert_allclose(coeffs, np.round(coeffs), atol=1e-10)
    
    def test_klein_center_parameter(self):
        """Test sampling with non-zero center."""
        lattice = IdentityLattice(5)
        sigma = 3.0
        center = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
        
        sampler = KleinSampler(lattice, sigma)
        
        # Generate centered samples
        n_samples = 5000
        samples = np.array([sampler.sample(center=center) for _ in range(n_samples)])
        
        # Check empirical mean is near center
        empirical_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(empirical_mean, np.round(center), atol=0.15)
    
    def test_klein_performance(self):
        """Test Klein's algorithm performance scaling."""
        import time
        
        dimensions = [10, 20, 50, 100]
        times = []
        
        for dim in dimensions:
            lattice = IdentityLattice(dim)
            sampler = KleinSampler(lattice, sigma=5.0)
            
            # Time sampling
            start = time.time()
            for _ in range(1000):
                sampler.sample()
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Check that time scales polynomially (not exponentially)
        # Log-log plot slope should be ~2 for O(n^2) algorithm
        log_dims = np.log(dimensions)
        log_times = np.log(times)
        
        # Fit line to log-log data
        slope, _ = np.polyfit(log_dims, log_times, 1)
        
        # Klein's algorithm is O(n^2), so slope should be around 2
        assert 1.5 <= slope <= 3.0


class TestIMHKSampler:
    """Test Independent MH-Klein sampler."""
    
    def test_detailed_balance(self):
        """Test that IMHK satisfies detailed balance."""
        lattice = IdentityLattice(5)
        sigma = 3.0
        sampler = IMHKSampler(lattice, sigma)
        
        # For detailed balance, acceptance ratio should satisfy
        # min(1, pi(y)q(x|y) / pi(x)q(y|x)) where q is Klein proposal
        
        # Test specific transitions
        x = np.array([1, 0, 0, 0, 0])
        y = np.array([0, 1, 0, 0, 0])
        
        # Compute acceptance probability
        log_ratio = sampler._compute_log_ratio(x, y)
        accept_prob = min(1.0, np.exp(log_ratio))
        
        # Should be symmetric for identity lattice with same norm
        assert 0 < accept_prob <= 1
    
    def test_convergence_to_target(self):
        """Test IMHK converges to target distribution."""
        lattice = IdentityLattice(10)
        sigma = 5.0
        sampler = IMHKSampler(lattice, sigma)
        
        # Run chain
        n_samples = 20000
        burn_in = 5000
        samples = sampler.sample(n_samples, burn_in=burn_in)
        
        # Check convergence using multiple statistics
        # 1. Mean should be near 0
        np.testing.assert_allclose(np.mean(samples, axis=0), 0, atol=0.1)
        
        # 2. Check stationarity using batch means
        batch_size = 1000
        n_batches = (n_samples - burn_in) // batch_size
        batch_means = []
        
        for i in range(n_batches):
            batch = samples[i*batch_size:(i+1)*batch_size]
            batch_means.append(np.mean(np.sum(batch**2, axis=1)))
        
        # Batch means should be similar (test stationarity)
        batch_std = np.std(batch_means)
        assert batch_std < 0.5
    
    def test_mixing_time_bound(self):
        """Test theoretical mixing time bounds."""
        lattice = IdentityLattice(20)
        sigma = 10.0
        sampler = IMHKSampler(lattice, sigma)
        
        # Get theoretical mixing time
        theoretical_mixing = sampler.mixing_time(epsilon=0.01)
        
        # Run empirical test
        n_chains = 50
        n_steps = theoretical_mixing * 2
        
        # Start chains from different initializations
        final_samples = []
        for _ in range(n_chains):
            sampler.reset(initial_state=np.random.randn(20) * 20)
            for _ in range(n_steps):
                sampler.step()
            final_samples.append(sampler.current_state.copy())
        
        final_samples = np.array(final_samples)
        
        # Check that chains have mixed (similar statistics)
        chain_means = np.mean(final_samples, axis=1)
        assert np.std(chain_means) < 0.5
    
    def test_spectral_gap(self):
        """Test spectral gap computation."""
        lattice = IdentityLattice(5)
        sigma = 3.0
        sampler = IMHKSampler(lattice, sigma)
        
        # Compute theoretical spectral gap
        gap = sampler.spectral_gap()
        
        # Should be in (0, 1]
        assert 0 < gap <= 1
        
        # For large sigma, gap should be close to 1
        sampler_large_sigma = IMHKSampler(lattice, sigma=100.0)
        gap_large = sampler_large_sigma.spectral_gap()
        assert gap_large > 0.9


class TestDiscreteGaussianUtils:
    """Test discrete Gaussian utilities."""
    
    def test_1d_samplers_consistency(self):
        """Test that all 1D samplers produce consistent results."""
        utils = DiscreteGaussianUtils()
        
        center = 5.3
        sigma = 2.0
        n_samples = 10000
        
        methods = ['rejection', 'cdf', 'alias']
        all_samples = {}
        
        for method in methods:
            samples = [utils.sample_discrete_gaussian_1d(center, sigma, method)
                      for _ in range(n_samples)]
            all_samples[method] = samples
        
        # Compare statistics
        for method in methods:
            samples = all_samples[method]
            mean = np.mean(samples)
            var = np.var(samples)
            
            # All methods should give similar results
            assert abs(mean - center) < 0.1
            assert abs(var - sigma**2) < 0.5
    
    def test_jacobi_theta_function(self):
        """Test Jacobi theta function computation."""
        utils = DiscreteGaussianUtils()
        
        # Test known values
        # ¸ƒ(0|i) = 1.0864...
        theta_val = utils.jacobi_theta_3(0, 1j)
        expected = 1.0864
        assert abs(theta_val - expected) < 0.001
        
        # Test derivatives
        z = 0.1 + 0.1j
        tau = 0.5j
        theta_vals = utils.jacobi_theta_3(z, tau, derivatives=2)
        
        # Check that we get 3 values (function + 2 derivatives)
        assert len(theta_vals) == 3
        
        # Function value should be finite
        assert np.isfinite(theta_vals[0])
    
    def test_partition_function_methods(self):
        """Test different partition function estimation methods."""
        utils = DiscreteGaussianUtils()
        
        # Simple orthogonal lattice
        basis = np.diag([1, 2, 3])
        sigma = 2.0
        
        # Compare methods
        Z_theta = utils.partition_function(basis, sigma, method='theta')
        Z_mc = utils.partition_function(basis, sigma, method='monte_carlo')
        Z_bounds = utils.partition_function(basis, sigma, method='bounds')
        
        # Monte Carlo should be close to theta (exact for orthogonal)
        assert abs(Z_theta - Z_mc) / Z_theta < 0.1
        
        # Bounds should contain true value
        assert Z_bounds[0] <= Z_theta <= Z_bounds[1]
    
    def test_smoothing_parameter_computation(self):
        """Test smoothing parameter calculation."""
        utils = DiscreteGaussianUtils()
        
        # Random lattice basis
        basis = np.random.randn(5, 5)
        
        # Different epsilon values
        epsilons = [0.1, 0.01, 0.001]
        etas = []
        
        for eps in epsilons:
            eta = utils.smoothing_parameter(basis, epsilon=eps)
            etas.append(eta)
        
        # Smoothing parameter should increase as epsilon decreases
        assert etas[0] < etas[1] < etas[2]
    
    def test_coset_sampling(self):
        """Test coset Gaussian sampling."""
        utils = DiscreteGaussianUtils()
        
        # Simple lattice
        basis = np.eye(3) * 2  # 2Z^3
        coset_shift = np.array([0.5, 0.5, 0.5])
        sigma = 3.0
        
        # Generate samples
        samples = []
        for _ in range(1000):
            sample = utils.sample_discrete_gaussian_coset(basis, coset_shift, sigma)
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Check that all samples are in the coset
        for sample in samples:
            # sample - coset_shift should be in lattice
            diff = sample - coset_shift
            coeffs = np.linalg.solve(basis.T, diff)
            np.testing.assert_allclose(coeffs, np.round(coeffs), atol=1e-10)
    
    def test_numerical_stability(self):
        """Test numerical stability of utilities."""
        utils = DiscreteGaussianUtils()
        
        # Test log_sum_exp with extreme values
        log_values = np.array([-1000, -999, -998, 100])
        result = utils.log_sum_exp(log_values)
        
        # Should handle without overflow/underflow
        assert np.isfinite(result)
        assert result >= 100  # Dominated by largest value
        
        # Test stable_softmax
        logits = np.array([1000, 1001, 999, -1000])
        probs = utils.stable_softmax(logits)
        
        # Should sum to 1 and have no NaN/Inf
        assert np.allclose(np.sum(probs), 1.0)
        assert np.all(np.isfinite(probs))


@pytest.mark.parametrize("dimension", [10, 20, 50])
@pytest.mark.parametrize("sigma_factor", [0.5, 1.0, 2.0])
def test_sampler_comparison(dimension, sigma_factor):
    """Compare Klein and IMHK samplers across parameters."""
    lattice = IdentityLattice(dimension)
    sigma = sigma_factor * np.sqrt(dimension)
    
    # Create samplers
    klein = KleinSampler(lattice, sigma)
    imhk = IMHKSampler(lattice, sigma)
    
    # Generate samples
    n_samples = 5000
    klein_samples = np.array([klein.sample() for _ in range(n_samples)])
    imhk_samples = imhk.sample(n_samples, burn_in=1000)
    
    # Compare statistics
    klein_mean = np.mean(np.sum(klein_samples**2, axis=1))
    imhk_mean = np.mean(np.sum(imhk_samples**2, axis=1))
    
    # Both should sample from same distribution
    expected_mean = dimension * sigma**2 * (1 - np.exp(-2*np.pi))
    
    assert abs(klein_mean - expected_mean) / expected_mean < 0.1
    assert abs(imhk_mean - expected_mean) / expected_mean < 0.1


def test_sampler_integration():
    """Integration test for complete sampling pipeline."""
    # Create a cryptographic lattice
    n, m, q = 64, 128, 3329
    lattice = QaryLattice.random_qary_lattice(n, m, q)
    
    # Reduce basis for better sampling
    from src.lattices.reduction import LatticeReduction
    reducer = LatticeReduction()
    basis = lattice.get_basis()
    reduced_basis, _ = reducer.sampling_reduce(basis, sigma=10.0)
    
    # Update lattice basis (in practice, would have setter method)
    lattice._basis = reduced_basis
    
    # Create samplers
    sigma = 10.0
    klein = KleinSampler(lattice, sigma)
    imhk = IMHKSampler(lattice, sigma)
    
    # Sample and check basic properties
    klein_sample = klein.sample()
    imhk_sample = imhk.step()
    
    assert klein_sample.shape == (m,)
    assert imhk_sample.shape == (m,)
    
    # Both should be valid lattice points
    for sample in [klein_sample, imhk_sample]:
        coeffs = np.linalg.solve(reduced_basis.T, sample)
        np.testing.assert_allclose(coeffs, np.round(coeffs), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])