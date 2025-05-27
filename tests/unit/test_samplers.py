"""
Unit tests for sampling algorithms.

Tests cover correctness, statistical properties, numerical accuracy,
and edge cases for Klein and IMHK sampling algorithms.
"""

import numpy as np
import pytest
from typing import List, Tuple
from scipy import stats
from unittest.mock import Mock, patch

from samplers.klein import KleinSampler
from samplers.imhk import IMHKSampler
from lattices.identity import IdentityLattice
from lattices.qary import QaryLattice
from samplers.utils import DiscreteGaussianUtils


class TestKleinSampler:
    """Test Klein's sampling algorithm."""
    
    def test_klein_sampler_initialization(self, identity_lattice_2d):
        """Test Klein sampler initialization."""
        sigma = 1.0
        sampler = KleinSampler(identity_lattice_2d, sigma)
        
        assert sampler.lattice == identity_lattice_2d
        assert sampler.sigma == sigma
        assert hasattr(sampler, 'Q')
        assert hasattr(sampler, 'R')
        
        # Check QR decomposition
        Q, R = sampler.Q, sampler.R
        basis = identity_lattice_2d.get_basis()
        reconstructed = Q @ R
        
        np.testing.assert_allclose(reconstructed, basis, rtol=1e-10)
    
    @pytest.mark.edge_case
    def test_klein_invalid_sigma(self, identity_lattice_2d):
        """Test Klein sampler with invalid sigma values."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            KleinSampler(identity_lattice_2d, sigma=0.0)
            
        with pytest.raises(ValueError, match="sigma must be positive"):
            KleinSampler(identity_lattice_2d, sigma=-1.0)
            
        with pytest.raises(ValueError, match="sigma must be finite"):
            KleinSampler(identity_lattice_2d, sigma=np.inf)
            
        with pytest.raises(ValueError, match="sigma must be finite"):
            KleinSampler(identity_lattice_2d, sigma=np.nan)
    
    @pytest.mark.statistical
    def test_klein_statistical_correctness_identity(self, statistical_config, tolerance_config):
        """Test statistical correctness of Klein sampling on identity lattice."""
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        sampler = KleinSampler(lattice, sigma)
        
        n_samples = statistical_config['n_samples']
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Test sample mean (should be close to zero)
        sample_mean = np.mean(samples, axis=0)
        expected_mean = np.zeros(2)
        
        # Statistical tolerance based on CLT
        std_error = sigma / np.sqrt(n_samples)
        tolerance = 3 * std_error
        
        np.testing.assert_allclose(
            sample_mean, expected_mean, atol=tolerance,
            err_msg="Klein sampler mean deviates from expected value"
        )
        
        # Test sample covariance (should approximate sigma^2 * I)
        sample_cov = np.cov(samples.T)
        expected_cov = sigma**2 * np.eye(2)
        
        # Allow for statistical fluctuation in covariance estimation
        cov_tolerance = statistical_config['statistical_rtol']
        np.testing.assert_allclose(
            sample_cov, expected_cov, rtol=cov_tolerance,
            err_msg="Klein sampler covariance deviates from expected value"
        )
    
    @pytest.mark.statistical  
    def test_klein_gaussian_profile(self, statistical_config, stat_utils):
        """Test that Klein samples follow discrete Gaussian distribution."""
        lattice = IdentityLattice(dimension=1)  # 1D for easier analysis
        sigma = 2.0
        sampler = KleinSampler(lattice, sigma)
        
        n_samples = statistical_config['n_samples']
        samples = np.array([sampler.sample()[0] for _ in range(n_samples)])
        
        # Compute empirical probabilities
        unique_values, counts = np.unique(samples, return_counts=True)
        empirical_probs = counts / n_samples
        
        # Compute theoretical probabilities for discrete Gaussian
        def discrete_gaussian_prob(x, sigma):
            """Theoretical probability for discrete Gaussian on Z."""
            normalization = np.sum([np.exp(-k**2 / (2 * sigma**2)) 
                                  for k in range(-100, 101)])
            return np.exp(-x**2 / (2 * sigma**2)) / normalization
        
        theoretical_probs = np.array([
            discrete_gaussian_prob(x, sigma) for x in unique_values
        ])
        
        # Chi-squared goodness of fit test
        # Only test values with sufficient expected counts
        expected_counts = theoretical_probs * n_samples
        valid_indices = expected_counts >= 5
        
        if np.sum(valid_indices) > 1:
            chi2_stat = np.sum(
                (counts[valid_indices] - expected_counts[valid_indices])**2 / 
                expected_counts[valid_indices]
            )
            
            # Critical value for alpha = 0.01
            dof = np.sum(valid_indices) - 1
            critical_value = stats.chi2.ppf(0.99, dof)
            
            assert chi2_stat < critical_value, \
                f"Klein samples fail chi-squared test: {chi2_stat} > {critical_value}"
    
    def test_klein_different_sigmas(self, identity_lattice_2d, statistical_config):
        """Test Klein sampling with different sigma values."""
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for sigma in sigmas:
            sampler = KleinSampler(identity_lattice_2d, sigma)
            
            # Generate fewer samples for this test
            samples = np.array([sampler.sample() for _ in range(1000)])
            sample_std = np.std(samples, axis=0)
            
            # Standard deviation should scale with sigma
            expected_std = sigma * np.ones(2)
            np.testing.assert_allclose(
                sample_std, expected_std, rtol=0.3,
                err_msg=f"Klein sampler std deviation incorrect for sigma={sigma}"
            )
    
    @pytest.mark.edge_case
    def test_klein_very_small_sigma(self, identity_lattice_2d):
        """Test Klein sampling with very small sigma."""
        sigma = 1e-6
        sampler = KleinSampler(identity_lattice_2d, sigma)
        
        # Generate samples
        samples = np.array([sampler.sample() for _ in range(100)])
        
        # Most samples should be at origin for very small sigma
        zero_samples = np.sum(np.all(samples == 0, axis=1))
        assert zero_samples > 80, \
            "Very small sigma should concentrate samples at origin"
    
    @pytest.mark.edge_case
    def test_klein_very_large_sigma(self, identity_lattice_2d):
        """Test Klein sampling with very large sigma."""
        sigma = 100.0
        sampler = KleinSampler(identity_lattice_2d, sigma)
        
        # Generate samples
        samples = np.array([sampler.sample() for _ in range(100)])
        
        # Samples should be spread out
        sample_range = np.max(samples, axis=0) - np.min(samples, axis=0)
        assert np.all(sample_range > 10), \
            "Large sigma should produce spread-out samples"
    
    def test_klein_lattice_point_validation(self, identity_lattice_2d):
        """Test that Klein sampler produces valid lattice points."""
        sigma = 1.0
        sampler = KleinSampler(identity_lattice_2d, sigma)
        
        for _ in range(100):
            sample = sampler.sample()
            
            # For identity lattice, all coordinates should be integers
            assert all(isinstance(x, (int, np.integer)) for x in sample)
            assert len(sample) == 2
    
    @pytest.mark.numerical
    def test_klein_qr_decomposition_accuracy(self, random_2d_basis, tolerance_config):
        """Test accuracy of QR decomposition in Klein sampler."""
        lattice = IdentityLattice(dimension=2)
        lattice.basis = random_2d_basis  # Override basis for testing
        
        sampler = KleinSampler(lattice, sigma=1.0)
        
        # Check QR decomposition
        Q, R = sampler.Q, sampler.R
        reconstructed = Q @ R
        
        np.testing.assert_allclose(
            reconstructed, random_2d_basis, 
            rtol=tolerance_config['rtol'],
            err_msg="QR decomposition in Klein sampler is inaccurate"
        )
        
        # Check orthogonality of Q
        QTQ = Q @ Q.T
        np.testing.assert_allclose(
            QTQ, np.eye(Q.shape[0]), 
            rtol=tolerance_config['rtol'],
            err_msg="Q matrix in Klein sampler is not orthogonal"
        )
    
    @pytest.mark.reproducibility
    def test_klein_reproducibility(self, identity_lattice_2d, test_seed):
        """Test reproducibility of Klein sampling."""
        sigma = 1.0
        
        # Generate samples with same seed
        np.random.seed(test_seed)
        sampler1 = KleinSampler(identity_lattice_2d, sigma)
        samples1 = [sampler1.sample() for _ in range(10)]
        
        np.random.seed(test_seed)
        sampler2 = KleinSampler(identity_lattice_2d, sigma)
        samples2 = [sampler2.sample() for _ in range(10)]
        
        # Samples should be identical
        for s1, s2 in zip(samples1, samples2):
            np.testing.assert_array_equal(s1, s2)


class TestIMHKSampler:
    """Test Independent Metropolis-Hastings-Klein (IMHK) algorithm."""
    
    def test_imhk_sampler_initialization(self, identity_lattice_2d):
        """Test IMHK sampler initialization."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        assert sampler.lattice == identity_lattice_2d
        assert sampler.sigma == sigma
        assert hasattr(sampler, 'proposal_sampler')
        assert isinstance(sampler.proposal_sampler, KleinSampler)
        assert sampler.current_sample is not None
    
    @pytest.mark.edge_case
    def test_imhk_invalid_parameters(self, identity_lattice_2d):
        """Test IMHK sampler with invalid parameters."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            IMHKSampler(identity_lattice_2d, sigma=0.0)
            
        with pytest.raises(ValueError, match="sigma must be positive"):
            IMHKSampler(identity_lattice_2d, sigma=-1.0)
    
    def test_imhk_proposal_mechanism(self, identity_lattice_2d):
        """Test IMHK proposal mechanism."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Test proposal generation
        proposal = sampler._generate_proposal()
        assert len(proposal) == 2
        assert all(isinstance(x, (int, np.integer)) for x in proposal)
    
    @pytest.mark.numerical
    def test_imhk_acceptance_ratio_computation(self, identity_lattice_2d, tolerance_config):
        """Test acceptance ratio computation in IMHK."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Test with known points
        current = np.array([0, 0])
        proposal = np.array([1, 0])
        
        # For identity lattice with sigma=1, can compute exact ratio
        ratio = sampler._compute_acceptance_ratio(current, proposal)
        
        assert 0 <= ratio <= 1, "Acceptance ratio must be between 0 and 1"
        assert np.isfinite(ratio), "Acceptance ratio must be finite"
        
        # Test symmetry property for identity lattice
        ratio_reverse = sampler._compute_acceptance_ratio(proposal, current)
        
        # Due to symmetry of identity lattice and Gaussian, ratios should be related
        # by the reverse probability ratio
        assert np.isfinite(ratio_reverse), "Reverse acceptance ratio must be finite"
    
    @pytest.mark.statistical
    def test_imhk_stationary_distribution(self, statistical_config, tolerance_config):
        """Test that IMHK converges to correct stationary distribution."""
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        sampler = IMHKSampler(lattice, sigma)
        
        # Run chain for burn-in
        burn_in = 1000
        for _ in range(burn_in):
            sampler.sample()
        
        # Collect samples
        n_samples = statistical_config['n_samples'] // 2  # Use fewer for MCMC
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Test sample mean
        sample_mean = np.mean(samples, axis=0)
        expected_mean = np.zeros(2)
        
        # More tolerance for MCMC due to correlation
        std_error = sigma / np.sqrt(n_samples // 10)  # Adjust for correlation
        tolerance = 5 * std_error
        
        np.testing.assert_allclose(
            sample_mean, expected_mean, atol=tolerance,
            err_msg="IMHK stationary distribution has incorrect mean"
        )
    
    def test_imhk_acceptance_rate(self, identity_lattice_2d, statistical_config):
        """Test IMHK acceptance rate is reasonable."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Track acceptance
        n_samples = 1000
        n_accepted = 0
        
        for _ in range(n_samples):
            old_sample = sampler.current_sample.copy()
            new_sample = sampler.sample()
            if not np.array_equal(old_sample, new_sample):
                n_accepted += 1
        
        acceptance_rate = n_accepted / n_samples
        
        # Acceptance rate should be reasonable (between 10% and 90%)
        assert 0.1 <= acceptance_rate <= 0.9, \
            f"IMHK acceptance rate {acceptance_rate} is outside reasonable range"
    
    @pytest.mark.statistical
    def test_imhk_autocorrelation(self, identity_lattice_2d, stat_utils):
        """Test autocorrelation properties of IMHK chain."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Generate chain
        n_samples = 5000
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Compute effective sample size for each dimension
        for dim in range(2):
            ess = stat_utils.effective_sample_size(samples[:, dim])
            
            # ESS should be reasonable fraction of total samples
            assert ess > n_samples * 0.01, \
                f"Effective sample size {ess} too low for dimension {dim}"
            assert ess <= n_samples, \
                f"Effective sample size {ess} exceeds total samples"
    
    def test_imhk_mixing_time_bound(self, identity_lattice_2d):
        """Test theoretical mixing time bound computation."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Test theoretical bound computation
        mixing_time = sampler.theoretical_mixing_time(epsilon=0.25)
        
        assert mixing_time > 0, "Mixing time must be positive"
        assert np.isfinite(mixing_time), "Mixing time must be finite"
        
        # For identity lattice, mixing time should be reasonable
        assert mixing_time < 1000, "Mixing time seems too large for identity lattice"
    
    def test_imhk_spectral_gap_bound(self, identity_lattice_2d, tolerance_config):
        """Test theoretical spectral gap bound computation."""
        sigma = 1.0
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Test spectral gap computation
        gap = sampler.theoretical_spectral_gap()
        
        assert 0 < gap <= 1, "Spectral gap must be in (0, 1]"
        assert np.isfinite(gap), "Spectral gap must be finite"
        
        # For well-conditioned lattices, gap should be bounded away from 0
        assert gap > 0.01, "Spectral gap too small for identity lattice"
    
    @pytest.mark.edge_case
    def test_imhk_pathological_lattice(self, pathological_basis):
        """Test IMHK with pathological lattice basis."""
        lattice = IdentityLattice(dimension=2)
        lattice.basis = pathological_basis  # Override for testing
        
        sigma = 1.0
        sampler = IMHKSampler(lattice, sigma)
        
        # Should still produce samples
        samples = [sampler.sample() for _ in range(10)]
        
        assert len(samples) == 10
        for sample in samples:
            assert len(sample) == 2
            assert all(np.isfinite(sample))
    
    @pytest.mark.reproducibility
    def test_imhk_reproducibility(self, identity_lattice_2d, test_seed):
        """Test reproducibility of IMHK sampling."""
        sigma = 1.0
        
        # Generate samples with same seed
        np.random.seed(test_seed)
        sampler1 = IMHKSampler(identity_lattice_2d, sigma)
        samples1 = [sampler1.sample() for _ in range(20)]
        
        np.random.seed(test_seed)
        sampler2 = IMHKSampler(identity_lattice_2d, sigma)
        samples2 = [sampler2.sample() for _ in range(20)]
        
        # Samples should be identical
        for s1, s2 in zip(samples1, samples2):
            np.testing.assert_array_equal(s1, s2)


class TestDiscreteGaussianUtils:
    """Test discrete Gaussian utility functions."""
    
    def test_discrete_gaussian_1d_sampling(self, statistical_config, tolerance_config):
        """Test 1D discrete Gaussian sampling."""
        utils = DiscreteGaussianUtils()
        sigma = 2.0
        
        # Generate samples
        n_samples = statistical_config['n_samples']
        samples = [utils.sample_discrete_gaussian_1d(sigma) for _ in range(n_samples)]
        
        # Test mean (should be close to 0)
        sample_mean = np.mean(samples)
        std_error = sigma / np.sqrt(n_samples)
        
        assert abs(sample_mean) < 3 * std_error, \
            "1D discrete Gaussian mean deviates from expected value"
        
        # Test that all samples are integers
        assert all(isinstance(x, (int, np.integer)) for x in samples)
    
    def test_jacobi_theta_function(self, tolerance_config):
        """Test Jacobi theta function computation."""
        utils = DiscreteGaussianUtils()
        
        # Test with known values
        z_values = [0.0, 0.5, 1.0, 2.0]
        tau_values = [0.5, 1.0, 2.0]
        
        for z in z_values:
            for tau in tau_values:
                theta = utils.jacobi_theta3(z, tau)
                
                assert np.isfinite(theta), f"Theta function not finite for z={z}, tau={tau}"
                assert theta.real > 0, f"Theta function not positive for z={z}, tau={tau}"
    
    def test_partition_function_computation(self, tolerance_config):
        """Test partition function computation."""
        utils = DiscreteGaussianUtils()
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        
        Z = utils.compute_partition_function(lattice, sigma)
        
        assert Z > 0, "Partition function must be positive"
        assert np.isfinite(Z), "Partition function must be finite"
        
        # For identity lattice, can compare with known approximation
        expected_Z_approx = (sigma * np.sqrt(2 * np.pi))**2
        
        # Should be in right ballpark (within factor of 2)
        assert 0.5 * expected_Z_approx < Z < 2.0 * expected_Z_approx, \
            "Partition function value seems incorrect"
    
    def test_discrete_gaussian_tail_bounds(self):
        """Test tail bound computation for discrete Gaussian."""
        utils = DiscreteGaussianUtils()
        sigma = 1.0
        
        # Test tail bound for different radii
        radii = [1, 2, 3, 5, 10]
        
        previous_bound = 1.0
        for r in radii:
            bound = utils.discrete_gaussian_tail_bound(r, sigma)
            
            assert 0 <= bound <= 1, f"Tail bound {bound} not in [0,1] for r={r}"
            assert bound <= previous_bound, \
                f"Tail bound not decreasing: {bound} > {previous_bound}"
            previous_bound = bound
    
    @pytest.mark.edge_case
    def test_discrete_gaussian_extreme_parameters(self):
        """Test discrete Gaussian with extreme parameters."""
        utils = DiscreteGaussianUtils()
        
        # Very small sigma
        sigma_small = 1e-3
        sample_small = utils.sample_discrete_gaussian_1d(sigma_small)
        assert isinstance(sample_small, (int, np.integer))
        assert abs(sample_small) <= 1, "Very small sigma should give samples near origin"
        
        # Large sigma
        sigma_large = 100.0
        sample_large = utils.sample_discrete_gaussian_1d(sigma_large)
        assert isinstance(sample_large, (int, np.integer))
        # Don't test range for large sigma as it can vary widely
    
    def test_special_lattice_samplers(self, tolerance_config):
        """Test special case samplers for specific lattices."""
        utils = DiscreteGaussianUtils()
        
        # Test Z^n sampler
        sigma = 1.0
        dimension = 3
        
        sample = utils.sample_integer_lattice(dimension, sigma)
        assert len(sample) == dimension
        assert all(isinstance(x, (int, np.integer)) for x in sample)
        
        # Test root lattice sampler (if implemented)
        if hasattr(utils, 'sample_root_lattice'):
            root_sample = utils.sample_root_lattice('A2', sigma)
            assert len(root_sample) >= 2  # A2 has dimension 2


@pytest.mark.integration
class TestSamplerIntegration:
    """Integration tests for samplers with different lattices."""
    
    def test_klein_on_qary_lattice(self, qary_lattice_small):
        """Test Klein sampling on q-ary lattice."""
        sigma = 1.0
        sampler = KleinSampler(qary_lattice_small, sigma)
        
        # Should produce samples
        samples = [sampler.sample() for _ in range(10)]
        
        for sample in samples:
            assert len(sample) == qary_lattice_small.get_dimension()
            assert all(np.isfinite(sample))
    
    def test_imhk_on_qary_lattice(self, qary_lattice_small):
        """Test IMHK sampling on q-ary lattice."""
        sigma = 1.0
        sampler = IMHKSampler(qary_lattice_small, sigma)
        
        # Should produce samples
        samples = [sampler.sample() for _ in range(10)]
        
        for sample in samples:
            assert len(sample) == qary_lattice_small.get_dimension()
            assert all(np.isfinite(sample))
    
    def test_sampler_consistency(self, identity_lattice_2d, statistical_config):
        """Test that Klein and IMHK produce statistically similar results."""
        sigma = 1.0
        klein_sampler = KleinSampler(identity_lattice_2d, sigma)
        imhk_sampler = IMHKSampler(identity_lattice_2d, sigma)
        
        # Generate samples
        n_samples = 2000  # Reduced for this comparison test
        
        klein_samples = np.array([klein_sampler.sample() for _ in range(n_samples)])
        
        # Burn-in for IMHK
        for _ in range(1000):
            imhk_sampler.sample()
        imhk_samples = np.array([imhk_sampler.sample() for _ in range(n_samples)])
        
        # Compare sample means
        klein_mean = np.mean(klein_samples, axis=0)
        imhk_mean = np.mean(imhk_samples, axis=0)
        
        # Should be similar within statistical tolerance
        np.testing.assert_allclose(
            klein_mean, imhk_mean, atol=0.2,
            err_msg="Klein and IMHK sample means differ significantly"
        )
        
        # Compare sample variances
        klein_var = np.var(klein_samples, axis=0)
        imhk_var = np.var(imhk_samples, axis=0)
        
        np.testing.assert_allclose(
            klein_var, imhk_var, rtol=0.3,
            err_msg="Klein and IMHK sample variances differ significantly"
        )