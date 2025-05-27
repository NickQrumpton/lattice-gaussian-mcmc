"""
Unit tests for diagnostic modules.

Tests cover convergence diagnostics, spectral analysis, and MCMC diagnostics
including statistical accuracy, numerical stability, and edge cases.
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Any
from unittest.mock import Mock, patch
from scipy import stats

from diagnostics.convergence import ConvergenceDiagnostics
from diagnostics.spectral import SpectralAnalysis
from diagnostics.mcmc import MCMCDiagnostics
from lattices.identity import IdentityLattice
from samplers.klein import KleinSampler
from samplers.imhk import IMHKSampler


class TestConvergenceDiagnostics:
    """Test convergence diagnostic methods."""
    
    def test_convergence_diagnostics_initialization(self):
        """Test ConvergenceDiagnostics initialization."""
        diagnostics = ConvergenceDiagnostics()
        
        assert hasattr(diagnostics, 'estimate_tvd')
        assert hasattr(diagnostics, 'effective_sample_size')
        assert hasattr(diagnostics, 'gelman_rubin_diagnostic')
        assert hasattr(diagnostics, 'geweke_diagnostic')
    
    @pytest.mark.statistical
    def test_tvd_estimation_known_distributions(self, statistical_config, tolerance_config):
        """Test TVD estimation between known distributions."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test TVD between identical distributions (should be ~0)
        n_samples = statistical_config['n_samples']
        samples1 = np.random.normal(0, 1, n_samples)
        samples2 = np.random.normal(0, 1, n_samples)
        
        tvd_identical = diagnostics.estimate_tvd(samples1, samples2)
        
        # TVD between samples from same distribution should be small
        assert 0 <= tvd_identical <= 1, "TVD must be between 0 and 1"
        assert tvd_identical < 0.1, "TVD between identical distributions should be small"
        
        # Test TVD between different distributions (should be larger)
        samples3 = np.random.normal(2, 1, n_samples)  # Different mean
        tvd_different = diagnostics.estimate_tvd(samples1, samples3)
        
        assert tvd_different > tvd_identical, "TVD should be larger for different distributions"
        assert tvd_different > 0.3, "TVD between different distributions should be substantial"
    
    def test_tvd_estimation_edge_cases(self):
        """Test TVD estimation edge cases."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with identical samples
        samples = np.array([1, 2, 3, 1, 2, 3])
        tvd_self = diagnostics.estimate_tvd(samples, samples)
        
        assert tvd_self == 0, "TVD of distribution with itself should be 0"
        
        # Test with single unique value
        constant_samples = np.ones(100)
        tvd_constant = diagnostics.estimate_tvd(constant_samples, constant_samples)
        
        assert tvd_constant == 0, "TVD of constant distribution with itself should be 0"
        
        # Test with empty arrays
        with pytest.raises((ValueError, IndexError)):
            diagnostics.estimate_tvd(np.array([]), np.array([]))
    
    @pytest.mark.statistical
    def test_effective_sample_size_computation(self, statistical_config, stat_utils):
        """Test effective sample size computation."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with independent samples (ESS should be close to n)
        n_samples = statistical_config['n_samples'] // 10  # Use fewer for speed
        independent_samples = np.random.randn(n_samples)
        
        ess_independent = diagnostics.effective_sample_size(independent_samples)
        
        assert ess_independent > 0, "ESS must be positive"
        assert ess_independent <= n_samples, "ESS cannot exceed number of samples"
        assert ess_independent > n_samples * 0.5, "ESS should be large for independent samples"
        
        # Test with highly correlated samples (ESS should be smaller)
        # Create AR(1) process with high correlation
        rho = 0.9
        correlated_samples = np.zeros(n_samples)
        correlated_samples[0] = np.random.randn()
        for i in range(1, n_samples):
            correlated_samples[i] = rho * correlated_samples[i-1] + np.sqrt(1-rho**2) * np.random.randn()
        
        ess_correlated = diagnostics.effective_sample_size(correlated_samples)
        
        assert ess_correlated < ess_independent, "ESS should be smaller for correlated samples"
        assert ess_correlated > 1, "ESS should be > 1 even for correlated samples"
    
    def test_effective_sample_size_edge_cases(self):
        """Test ESS computation edge cases."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with constant series
        constant_series = np.ones(100)
        ess_constant = diagnostics.effective_sample_size(constant_series)
        
        # ESS should be small for constant series
        assert ess_constant <= 1.1, "ESS should be ~1 for constant series"
        
        # Test with short series
        short_series = np.array([1, 2, 3])
        ess_short = diagnostics.effective_sample_size(short_series)
        
        assert ess_short > 0, "ESS must be positive for short series"
        assert ess_short <= 3, "ESS cannot exceed series length"
    
    @pytest.mark.statistical
    def test_gelman_rubin_diagnostic(self, statistical_config):
        """Test Gelman-Rubin diagnostic (R-hat)."""
        diagnostics = ConvergenceDiagnostics()
        
        # Create multiple chains from same distribution
        n_chains = 4
        n_samples = statistical_config['n_samples'] // 10
        
        # Converged chains (should have R-hat ≈ 1)
        converged_chains = [
            np.random.normal(0, 1, n_samples) for _ in range(n_chains)
        ]
        
        rhat_converged = diagnostics.gelman_rubin_diagnostic(converged_chains)
        
        assert rhat_converged >= 1.0, "R-hat must be ≥ 1"
        assert rhat_converged < 1.2, "R-hat should be close to 1 for converged chains"
        
        # Non-converged chains (should have R-hat > 1)
        non_converged_chains = [
            np.random.normal(i, 1, n_samples) for i in range(n_chains)
        ]
        
        rhat_non_converged = diagnostics.gelman_rubin_diagnostic(non_converged_chains)
        
        assert rhat_non_converged > rhat_converged, "R-hat should be larger for non-converged chains"
        assert rhat_non_converged > 1.1, "R-hat should be > 1.1 for clearly non-converged chains"
    
    def test_gelman_rubin_edge_cases(self):
        """Test Gelman-Rubin diagnostic edge cases."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with single chain
        with pytest.raises(ValueError, match="Need at least 2 chains"):
            diagnostics.gelman_rubin_diagnostic([np.random.randn(100)])
        
        # Test with chains of different lengths
        chains_diff_length = [
            np.random.randn(100),
            np.random.randn(50)
        ]
        
        with pytest.raises(ValueError, match="All chains must have same length"):
            diagnostics.gelman_rubin_diagnostic(chains_diff_length)
        
        # Test with constant chains
        constant_chains = [
            np.ones(100),
            np.ones(100)
        ]
        
        # Should handle gracefully (R-hat = 1 or NaN)
        rhat_constant = diagnostics.gelman_rubin_diagnostic(constant_chains)
        assert np.isnan(rhat_constant) or np.isclose(rhat_constant, 1.0), \
            "R-hat for constant chains should be 1 or NaN"
    
    @pytest.mark.statistical
    def test_geweke_diagnostic(self, statistical_config):
        """Test Geweke diagnostic for convergence."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with stationary chain
        n_samples = statistical_config['n_samples'] // 5
        stationary_chain = np.random.randn(n_samples)
        
        z_score = diagnostics.geweke_diagnostic(stationary_chain)
        
        # Z-score should be approximately standard normal
        assert np.abs(z_score) < 3, "Geweke z-score should be < 3 for stationary chain"
        
        # Test with non-stationary chain (trend)
        trend_chain = np.cumsum(np.random.randn(n_samples)) / np.sqrt(np.arange(1, n_samples + 1))
        z_score_trend = diagnostics.geweke_diagnostic(trend_chain)
        
        # Should detect non-stationarity (higher z-score)
        assert np.abs(z_score_trend) >= np.abs(z_score), \
            "Geweke should detect trend in non-stationary chain"
    
    def test_geweke_diagnostic_edge_cases(self):
        """Test Geweke diagnostic edge cases."""
        diagnostics = ConvergenceDiagnostics()
        
        # Test with short chain
        short_chain = np.random.randn(20)
        
        with pytest.raises(ValueError, match="Chain too short"):
            diagnostics.geweke_diagnostic(short_chain)
        
        # Test with constant chain
        constant_chain = np.ones(1000)
        
        # Should handle gracefully (z-score = 0 or NaN)
        z_score_constant = diagnostics.geweke_diagnostic(constant_chain)
        assert np.isnan(z_score_constant) or np.isclose(z_score_constant, 0.0), \
            "Geweke z-score for constant chain should be 0 or NaN"
    
    @pytest.mark.integration
    def test_convergence_on_mcmc_chains(self, identity_lattice_2d):
        """Test convergence diagnostics on actual MCMC chains."""
        diagnostics = ConvergenceDiagnostics()
        
        # Generate MCMC chains using IMHK sampler
        sigma = 1.0
        n_chains = 3
        n_samples = 2000
        
        chains = []
        for _ in range(n_chains):
            sampler = IMHKSampler(identity_lattice_2d, sigma)
            
            # Burn-in
            for _ in range(500):
                sampler.sample()
            
            # Collect samples
            chain = np.array([sampler.sample()[0] for _ in range(n_samples)])  # Just first coordinate
            chains.append(chain)
        
        # Test Gelman-Rubin diagnostic
        rhat = diagnostics.gelman_rubin_diagnostic(chains)
        assert 1.0 <= rhat < 1.3, f"MCMC chains should converge: R-hat = {rhat}"
        
        # Test effective sample size
        ess = diagnostics.effective_sample_size(chains[0])
        assert ess > 100, f"ESS should be reasonable: ESS = {ess}"
        
        # Test Geweke diagnostic
        z_score = diagnostics.geweke_diagnostic(chains[0])
        assert np.abs(z_score) < 3, f"Geweke diagnostic should pass: z = {z_score}"


class TestSpectralAnalysis:
    """Test spectral gap analysis methods."""
    
    def test_spectral_analysis_initialization(self):
        """Test SpectralAnalysis initialization."""
        analysis = SpectralAnalysis()
        
        assert hasattr(analysis, 'compute_spectral_gap')
        assert hasattr(analysis, 'estimate_mixing_time')
        assert hasattr(analysis, 'compute_eigenvalues')
        assert hasattr(analysis, 'theoretical_bounds')
    
    @pytest.mark.numerical
    def test_spectral_gap_computation_identity(self, identity_lattice_2d, tolerance_config):
        """Test spectral gap computation for identity lattice."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        # Compute empirical spectral gap
        gap_empirical = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='empirical')
        
        assert 0 < gap_empirical <= 1, "Spectral gap must be in (0, 1]"
        assert np.isfinite(gap_empirical), "Spectral gap must be finite"
        
        # Compute theoretical spectral gap
        gap_theoretical = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='theoretical')
        
        assert 0 < gap_theoretical <= 1, "Theoretical spectral gap must be in (0, 1]"
        assert np.isfinite(gap_theoretical), "Theoretical spectral gap must be finite"
        
        # Empirical and theoretical should be reasonably close
        np.testing.assert_allclose(
            gap_empirical, gap_theoretical, rtol=0.5,
            err_msg="Empirical and theoretical spectral gaps differ significantly"
        )
    
    def test_spectral_gap_different_sigmas(self, identity_lattice_2d):
        """Test spectral gap with different sigma values."""
        analysis = SpectralAnalysis()
        sigmas = [0.5, 1.0, 2.0, 5.0]
        
        gaps = []
        for sigma in sigmas:
            gap = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='theoretical')
            gaps.append(gap)
            
            assert 0 < gap <= 1, f"Invalid spectral gap for sigma={sigma}: {gap}"
        
        # Generally, larger sigma should give smaller gap (slower mixing)
        # But this relationship might not be monotonic for all lattices
        assert all(0 < g <= 1 for g in gaps), "All spectral gaps should be valid"
    
    @pytest.mark.edge_case
    def test_spectral_gap_edge_cases(self, identity_lattice_2d):
        """Test spectral gap computation edge cases."""
        analysis = SpectralAnalysis()
        
        # Very small sigma
        gap_small = analysis.compute_spectral_gap(identity_lattice_2d, sigma=1e-6, method='theoretical')
        assert 0 < gap_small <= 1, "Spectral gap invalid for very small sigma"
        
        # Very large sigma
        gap_large = analysis.compute_spectral_gap(identity_lattice_2d, sigma=100.0, method='theoretical')
        assert 0 < gap_large <= 1, "Spectral gap invalid for very large sigma"
        
        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            analysis.compute_spectral_gap(identity_lattice_2d, sigma=1.0, method='invalid')
    
    def test_eigenvalue_computation(self, identity_lattice_2d, tolerance_config):
        """Test eigenvalue computation of transition matrix."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        eigenvalues = analysis.compute_eigenvalues(identity_lattice_2d, sigma, n_eigenvalues=10)
        
        # Check basic properties
        assert len(eigenvalues) <= 10, "Should not return more eigenvalues than requested"
        assert all(np.isfinite(eigenvalues)), "All eigenvalues should be finite"
        assert all(0 <= ev <= 1 for ev in eigenvalues), "Eigenvalues should be in [0, 1]"
        
        # Largest eigenvalue should be 1 (stochastic matrix)
        assert np.isclose(eigenvalues[0], 1.0, rtol=tolerance_config['rtol']), \
            "Largest eigenvalue should be 1"
        
        # Eigenvalues should be in descending order
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i+1] - tolerance_config['rtol'], \
                "Eigenvalues should be in descending order"
    
    @pytest.mark.numerical
    def test_mixing_time_estimation(self, identity_lattice_2d, tolerance_config):
        """Test mixing time estimation."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        epsilon = 0.25
        
        # Compute mixing time from spectral gap
        mixing_time = analysis.estimate_mixing_time(identity_lattice_2d, sigma, epsilon)
        
        assert mixing_time > 0, "Mixing time must be positive"
        assert np.isfinite(mixing_time), "Mixing time must be finite"
        assert mixing_time < 1e6, "Mixing time seems unreasonably large"
        
        # Test with different epsilon values
        epsilons = [0.1, 0.25, 0.5]
        mixing_times = []
        
        for eps in epsilons:
            mt = analysis.estimate_mixing_time(identity_lattice_2d, sigma, eps)
            mixing_times.append(mt)
        
        # Smaller epsilon should give larger mixing time
        for i in range(len(mixing_times) - 1):
            assert mixing_times[i] >= mixing_times[i+1] * 0.9, \
                "Mixing time should decrease with larger epsilon"
    
    def test_theoretical_bounds(self, identity_lattice_2d):
        """Test theoretical bound computations."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        bounds = analysis.theoretical_bounds(identity_lattice_2d, sigma)
        
        assert isinstance(bounds, dict), "Bounds should be returned as dictionary"
        
        expected_keys = ['spectral_gap_lower', 'spectral_gap_upper', 'mixing_time_upper']
        for key in expected_keys:
            assert key in bounds, f"Missing bound: {key}"
            assert bounds[key] > 0, f"Bound {key} must be positive"
            assert np.isfinite(bounds[key]), f"Bound {key} must be finite"
        
        # Check bound relationships
        assert bounds['spectral_gap_lower'] <= bounds['spectral_gap_upper'], \
            "Spectral gap bounds should be ordered correctly"
    
    @pytest.mark.integration
    def test_spectral_analysis_mcmc_consistency(self, identity_lattice_2d):
        """Test consistency between spectral analysis and MCMC behavior."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        # Compute theoretical mixing time
        theoretical_mixing = analysis.estimate_mixing_time(identity_lattice_2d, sigma, epsilon=0.25)
        
        # Run MCMC and estimate empirical mixing time
        sampler = IMHKSampler(identity_lattice_2d, sigma)
        n_samples = 5000
        
        # Generate chain
        samples = np.array([sampler.sample()[0] for _ in range(n_samples)])
        
        # Estimate empirical mixing time using autocorrelation
        def autocorr_time(chain, c=5):
            # Simplified autocorrelation time estimation
            autocorr = np.correlate(chain - np.mean(chain), chain - np.mean(chain), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first point where autocorr < 1/e
            threshold = 1.0 / np.e
            below_threshold = np.where(autocorr < threshold)[0]
            if len(below_threshold) > 0:
                return below_threshold[0]
            else:
                return len(autocorr) - 1
        
        empirical_mixing = autocorr_time(samples)
        
        # Theoretical and empirical should be in same ballpark
        ratio = theoretical_mixing / empirical_mixing
        assert 0.1 < ratio < 10, \
            f"Theoretical and empirical mixing times differ too much: {theoretical_mixing} vs {empirical_mixing}"


class TestMCMCDiagnostics:
    """Test comprehensive MCMC diagnostic methods."""
    
    def test_mcmc_diagnostics_initialization(self):
        """Test MCMCDiagnostics initialization."""
        diagnostics = MCMCDiagnostics()
        
        assert hasattr(diagnostics, 'comprehensive_analysis')
        assert hasattr(diagnostics, 'autocorrelation_analysis')
        assert hasattr(diagnostics, 'trace_analysis')
        assert hasattr(diagnostics, 'acceptance_rate_analysis')
    
    def test_autocorrelation_analysis(self, statistical_config):
        """Test autocorrelation function analysis."""
        diagnostics = MCMCDiagnostics()
        
        # Test with independent samples
        n_samples = statistical_config['n_samples'] // 5
        independent_samples = np.random.randn(n_samples)
        
        autocorr_result = diagnostics.autocorrelation_analysis(
            independent_samples, max_lag=50
        )
        
        assert isinstance(autocorr_result, dict), "Autocorrelation result should be dict"
        assert 'autocorrelation' in autocorr_result
        assert 'integrated_time' in autocorr_result
        assert 'effective_sample_size' in autocorr_result
        
        # For independent samples, integrated time should be small
        assert autocorr_result['integrated_time'] < 10, \
            "Integrated time should be small for independent samples"
        
        # ESS should be close to sample size
        assert autocorr_result['effective_sample_size'] > n_samples * 0.5, \
            "ESS should be large for independent samples"
    
    def test_trace_analysis(self, statistical_config):
        """Test trace plot analysis."""
        diagnostics = MCMCDiagnostics()
        
        # Test with stationary chain
        n_samples = statistical_config['n_samples'] // 10
        stationary_chain = np.random.randn(n_samples)
        
        trace_result = diagnostics.trace_analysis(stationary_chain)
        
        assert isinstance(trace_result, dict), "Trace result should be dict"
        assert 'mean_stability' in trace_result
        assert 'variance_stability' in trace_result
        assert 'trend_test' in trace_result
        
        # For stationary chain, should be stable
        assert trace_result['mean_stability'], "Mean should be stable for stationary chain"
        assert trace_result['variance_stability'], "Variance should be stable for stationary chain"
        assert not trace_result['trend_test']['significant_trend'], \
            "Should not detect trend in stationary chain"
    
    def test_acceptance_rate_analysis(self):
        """Test acceptance rate analysis for MCMC chains."""
        diagnostics = MCMCDiagnostics()
        
        # Test with different acceptance rates
        acceptance_rates = [0.2, 0.4, 0.6, 0.8]
        
        for rate in acceptance_rates:
            analysis = diagnostics.acceptance_rate_analysis(rate)
            
            assert isinstance(analysis, dict), "Acceptance rate analysis should be dict"
            assert 'acceptance_rate' in analysis
            assert 'assessment' in analysis
            assert 'recommendation' in analysis
            
            assert analysis['acceptance_rate'] == rate
            assert isinstance(analysis['assessment'], str)
            assert isinstance(analysis['recommendation'], str)
    
    @pytest.mark.integration
    def test_comprehensive_mcmc_analysis(self, identity_lattice_2d):
        """Test comprehensive MCMC analysis on real chains."""
        diagnostics = MCMCDiagnostics()
        
        # Generate MCMC chains
        sigma = 1.0
        n_chains = 3
        n_samples = 2000
        
        all_chains = []
        acceptance_rates = []
        
        for _ in range(n_chains):
            sampler = IMHKSampler(identity_lattice_2d, sigma)
            
            # Track acceptances
            n_accepted = 0
            chain = []
            
            # Burn-in
            for _ in range(500):
                sampler.sample()
            
            # Collect samples and track acceptance
            for _ in range(n_samples):
                old_sample = sampler.current_sample.copy()
                new_sample = sampler.sample()
                chain.append(new_sample[0])  # Just first coordinate
                
                if not np.array_equal(old_sample, new_sample):
                    n_accepted += 1
            
            all_chains.append(np.array(chain))
            acceptance_rates.append(n_accepted / n_samples)
        
        # Run comprehensive analysis
        comprehensive_result = diagnostics.comprehensive_analysis(
            all_chains, acceptance_rates
        )
        
        assert isinstance(comprehensive_result, dict), "Comprehensive result should be dict"
        
        # Check main sections
        expected_sections = [
            'gelman_rubin', 'effective_sample_sizes', 'autocorrelation_times',
            'acceptance_rate_analysis', 'convergence_assessment'
        ]
        
        for section in expected_sections:
            assert section in comprehensive_result, f"Missing section: {section}"
        
        # Check convergence assessment
        convergence = comprehensive_result['convergence_assessment']
        assert isinstance(convergence['converged'], bool)
        assert isinstance(convergence['summary'], str)
        assert 'recommendations' in convergence
    
    def test_diagnostic_edge_cases(self):
        """Test MCMC diagnostics edge cases."""
        diagnostics = MCMCDiagnostics()
        
        # Test with constant chain
        constant_chain = np.ones(1000)
        
        trace_result = diagnostics.trace_analysis(constant_chain)
        assert not trace_result['variance_stability'], \
            "Constant chain should fail variance stability test"
        
        # Test with very short chain
        short_chain = np.random.randn(10)
        
        with pytest.raises(ValueError, match="Chain too short"):
            diagnostics.autocorrelation_analysis(short_chain, max_lag=50)
        
        # Test with extreme acceptance rates
        extreme_rates = [0.0, 1.0]
        for rate in extreme_rates:
            analysis = diagnostics.acceptance_rate_analysis(rate)
            assert 'poor' in analysis['assessment'].lower() or 'extreme' in analysis['assessment'].lower()


@pytest.mark.reproducibility
class TestDiagnosticReproducibility:
    """Test reproducibility of diagnostic computations."""
    
    def test_convergence_diagnostic_reproducibility(self, test_seed, statistical_config):
        """Test that convergence diagnostics are reproducible."""
        diagnostics = ConvergenceDiagnostics()
        
        # Generate deterministic chain
        np.random.seed(test_seed)
        chain1 = np.random.randn(statistical_config['n_samples'] // 10)
        
        np.random.seed(test_seed)
        chain2 = np.random.randn(statistical_config['n_samples'] // 10)
        
        # Compute diagnostics
        ess1 = diagnostics.effective_sample_size(chain1)
        ess2 = diagnostics.effective_sample_size(chain2)
        
        assert np.isclose(ess1, ess2, rtol=1e-10), "ESS computation should be reproducible"
        
        # Test Geweke diagnostic
        geweke1 = diagnostics.geweke_diagnostic(chain1)
        geweke2 = diagnostics.geweke_diagnostic(chain2)
        
        assert np.isclose(geweke1, geweke2, rtol=1e-10), "Geweke diagnostic should be reproducible"
    
    def test_spectral_analysis_reproducibility(self, identity_lattice_2d, test_seed):
        """Test that spectral analysis is reproducible."""
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        # Compute spectral gap twice with same seed
        np.random.seed(test_seed)
        gap1 = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='theoretical')
        
        np.random.seed(test_seed)
        gap2 = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='theoretical')
        
        assert np.isclose(gap1, gap2, rtol=1e-10), "Spectral gap computation should be reproducible"


@pytest.mark.performance
class TestDiagnosticPerformance:
    """Test performance of diagnostic computations."""
    
    def test_diagnostic_computation_speed(self, performance_config):
        """Test that diagnostic computations complete in reasonable time."""
        import time
        
        diagnostics = ConvergenceDiagnostics()
        
        # Test with large chain
        large_chain = np.random.randn(50000)
        
        # Test ESS computation speed
        start_time = time.time()
        ess = diagnostics.effective_sample_size(large_chain)
        ess_time = time.time() - start_time
        
        assert ess_time < performance_config['max_time_simple'], \
            f"ESS computation too slow: {ess_time:.2f}s"
        
        # Test Geweke diagnostic speed
        start_time = time.time()
        geweke = diagnostics.geweke_diagnostic(large_chain)
        geweke_time = time.time() - start_time
        
        assert geweke_time < performance_config['max_time_simple'], \
            f"Geweke diagnostic too slow: {geweke_time:.2f}s"
    
    def test_spectral_analysis_performance(self, identity_lattice_2d, performance_config):
        """Test spectral analysis performance."""
        import time
        
        analysis = SpectralAnalysis()
        sigma = 1.0
        
        # Test theoretical spectral gap computation
        start_time = time.time()
        gap = analysis.compute_spectral_gap(identity_lattice_2d, sigma, method='theoretical')
        gap_time = time.time() - start_time
        
        assert gap_time < performance_config['max_time_simple'], \
            f"Spectral gap computation too slow: {gap_time:.2f}s"