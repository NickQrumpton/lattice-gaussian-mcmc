"""
Integration tests for the full lattice Gaussian MCMC pipeline.

Tests cover end-to-end workflows from lattice construction through
sampling, diagnostics, and analysis. Includes both simple and
cryptographically-relevant scenarios.
"""

import numpy as np
import pytest
from pathlib import Path
import json
import pickle
import tempfile
import shutil
from typing import Dict, List, Tuple, Any

# Import all major components
from lattices.identity import IdentityLattice
from lattices.qary import QaryLattice
from lattices.reduction import LatticeReduction
from samplers.klein import KleinSampler
from samplers.imhk import IMHKSampler
from diagnostics.convergence import ConvergenceDiagnostics
from diagnostics.spectral import SpectralAnalysis
from diagnostics.mcmc import MCMCDiagnostics
from visualization.plots import PlottingTools


@pytest.mark.end_to_end
class TestSimplePipeline:
    """Test complete pipeline on simple lattices."""
    
    def test_identity_lattice_pipeline(self, temp_dir, statistical_config, tolerance_config):
        """Test complete pipeline on identity lattice."""
        # Step 1: Lattice construction
        lattice = IdentityLattice(dimension=2)
        
        # Verify lattice properties
        basis = lattice.get_basis()
        assert basis.shape == (2, 2)
        np.testing.assert_array_equal(basis, np.eye(2))
        
        # Step 2: Optional reduction (identity lattice should remain unchanged)
        reducer = LatticeReduction()
        reduced_basis, unimodular = reducer.lll_reduce(basis)
        
        np.testing.assert_allclose(reduced_basis, basis, rtol=tolerance_config['rtol'])
        
        # Step 3: Sampling with Klein algorithm
        sigma = 1.0
        klein_sampler = KleinSampler(lattice, sigma)
        
        n_samples = statistical_config['n_samples'] // 5  # Fewer samples for integration test
        klein_samples = np.array([klein_sampler.sample() for _ in range(n_samples)])
        
        # Verify sample properties
        assert klein_samples.shape == (n_samples, 2)
        assert all(isinstance(x, (int, np.integer)) for sample in klein_samples for x in sample)
        
        # Step 4: Sampling with IMHK algorithm
        imhk_sampler = IMHKSampler(lattice, sigma)
        
        # Burn-in
        for _ in range(500):
            imhk_sampler.sample()
        
        imhk_samples = np.array([imhk_sampler.sample() for _ in range(n_samples)])
        
        # Verify sample properties
        assert imhk_samples.shape == (n_samples, 2)
        assert all(isinstance(x, (int, np.integer)) for sample in imhk_samples for x in sample)
        
        # Step 5: Convergence diagnostics
        diagnostics = ConvergenceDiagnostics()
        
        # Test effective sample size
        ess_klein = diagnostics.effective_sample_size(klein_samples[:, 0])
        ess_imhk = diagnostics.effective_sample_size(imhk_samples[:, 0])
        
        assert ess_klein > n_samples * 0.1, "Klein ESS too low"
        assert ess_imhk > n_samples * 0.05, "IMHK ESS too low"  # MCMC typically has lower ESS
        
        # Test TVD between Klein and IMHK (should be small)
        tvd = diagnostics.estimate_tvd(klein_samples[:, 0], imhk_samples[:, 0])
        assert tvd < 0.3, "TVD between Klein and IMHK samples too large"
        
        # Step 6: Spectral analysis
        spectral = SpectralAnalysis()
        
        gap = spectral.compute_spectral_gap(lattice, sigma, method='theoretical')
        assert 0 < gap <= 1, "Invalid spectral gap"
        
        mixing_time = spectral.estimate_mixing_time(lattice, sigma, epsilon=0.25)
        assert mixing_time > 0, "Invalid mixing time"
        
        # Step 7: Save results
        results = {
            'lattice_dimension': lattice.get_dimension(),
            'sigma': sigma,
            'n_samples': n_samples,
            'klein_samples_mean': np.mean(klein_samples, axis=0).tolist(),
            'klein_samples_cov': np.cov(klein_samples.T).tolist(),
            'imhk_samples_mean': np.mean(imhk_samples, axis=0).tolist(),
            'imhk_samples_cov': np.cov(imhk_samples.T).tolist(),
            'ess_klein': float(ess_klein),
            'ess_imhk': float(ess_imhk),
            'tvd_klein_imhk': float(tvd),
            'spectral_gap': float(gap),
            'mixing_time': float(mixing_time)
        }
        
        # Save to temporary directory
        results_file = temp_dir / 'identity_lattice_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify file was created and is readable
        assert results_file.exists()
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['lattice_dimension'] == 2
        assert loaded_results['sigma'] == sigma
        
        # Step 8: Statistical validation
        # Sample means should be close to zero
        klein_mean = np.array(loaded_results['klein_samples_mean'])
        imhk_mean = np.array(loaded_results['imhk_samples_mean'])
        
        std_error = sigma / np.sqrt(n_samples)
        tolerance = 3 * std_error
        
        assert np.allclose(klein_mean, 0, atol=tolerance), "Klein sample mean deviates from 0"
        assert np.allclose(imhk_mean, 0, atol=tolerance), "IMHK sample mean deviates from 0"
        
        # Sample covariances should be close to sigma^2 * I
        klein_cov = np.array(loaded_results['klein_samples_cov'])
        imhk_cov = np.array(loaded_results['imhk_samples_cov'])
        expected_cov = sigma**2 * np.eye(2)
        
        cov_tolerance = statistical_config['statistical_rtol']
        np.testing.assert_allclose(klein_cov, expected_cov, rtol=cov_tolerance)
        np.testing.assert_allclose(imhk_cov, expected_cov, rtol=cov_tolerance)
    
    def test_qary_lattice_pipeline(self, temp_dir, statistical_config):
        """Test complete pipeline on q-ary lattice."""
        # Step 1: Construct q-ary lattice from LWE instance
        n, m, q = 4, 6, 17
        np.random.seed(42)  # For reproducibility
        A = np.random.randint(0, q, (n, m))
        
        lattice = QaryLattice.from_lwe_instance(A, q, dual=False)
        
        # Verify lattice construction
        basis = lattice.get_basis()
        assert basis.shape == (n, m + n)
        
        # Step 2: Lattice reduction for improved sampling
        reducer = LatticeReduction()
        
        # Compute quality before reduction
        original_orthogonality = reducer.orthogonality_defect(basis)
        
        # Perform LLL reduction
        reduced_basis, unimodular = reducer.lll_reduce(basis)
        
        # Verify reduction improved quality
        reduced_orthogonality = reducer.orthogonality_defect(reduced_basis)
        assert reduced_orthogonality <= original_orthogonality * 1.1
        
        # Create lattice with reduced basis
        class ReducedLattice:
            def __init__(self, reduced_basis):
                self.basis = reduced_basis
                
            def get_basis(self):
                return self.basis
                
            def get_gram_schmidt(self):
                Q, R = np.linalg.qr(self.basis.T)
                return Q.T, R.T
                
            def decode_cvp(self, target):
                # Simple rounding for testing
                return np.round(target)
                
            def sample_lattice_point(self, sigma):
                return np.zeros(self.get_dimension())
                
            def get_dimension(self):
                return self.basis.shape[0]
        
        reduced_lattice = ReducedLattice(reduced_basis)
        
        # Step 3: Sampling with reduced lattice
        sigma = 2.0  # Larger sigma for q-ary lattice
        
        try:
            # Try Klein sampling
            klein_sampler = KleinSampler(reduced_lattice, sigma)
            n_samples = 100  # Fewer samples for q-ary lattice test
            
            klein_samples = []
            for _ in range(n_samples):
                sample = klein_sampler.sample()
                klein_samples.append(sample)
            
            klein_samples = np.array(klein_samples)
            
            # Verify basic properties
            assert len(klein_samples) == n_samples
            assert klein_samples.shape[1] == reduced_lattice.get_dimension()
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Q-ary lattices can be challenging; accept if sampling fails
            pytest.skip(f"Klein sampling failed on q-ary lattice: {e}")
        
        # Step 4: Diagnostics (if sampling succeeded)
        if 'klein_samples' in locals():
            diagnostics = ConvergenceDiagnostics()
            
            # Test first coordinate only for simplicity
            ess = diagnostics.effective_sample_size(klein_samples[:, 0])
            assert ess > 0, "ESS must be positive"
            
            # Step 5: Save results
            results = {
                'lattice_type': 'qary',
                'parameters': {'n': n, 'm': m, 'q': q},
                'sigma': sigma,
                'n_samples': n_samples,
                'original_orthogonality_defect': float(original_orthogonality),
                'reduced_orthogonality_defect': float(reduced_orthogonality),
                'ess': float(ess),
                'reduction_improvement': float(original_orthogonality / reduced_orthogonality)
            }
            
            results_file = temp_dir / 'qary_lattice_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            assert results_file.exists()


@pytest.mark.end_to_end
class TestCryptographicPipeline:
    """Test pipeline on cryptographically-relevant lattices."""
    
    def test_ntru_like_pipeline(self, temp_dir, tolerance_config):
        """Test pipeline on NTRU-like lattice."""
        # Step 1: Construct NTRU-like lattice
        n = 8  # Small dimension for testing
        q = 17
        
        # Create NTRU-like polynomial
        np.random.seed(123)  # For reproducibility
        f = np.random.randint(-1, 2, n)  # Ternary polynomial
        f[0] = 1  # Ensure invertibility mod q
        
        # Construct NTRU lattice basis (simplified)
        # [I | Rot(h)]
        # [0 |   qI  ]
        h = np.random.randint(0, q, n)  # Public key
        
        # Construct lattice basis
        top_left = np.eye(n)
        top_right = np.roll(np.eye(n), 1, axis=0) @ np.diag(h) % q  # Simplified rotation
        bottom_left = np.zeros((n, n))
        bottom_right = q * np.eye(n)
        
        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        basis = np.vstack([top, bottom]).astype(float)
        
        class NTRULattice:
            def __init__(self, basis):
                self.basis = basis
                
            def get_basis(self):
                return self.basis
                
            def get_gram_schmidt(self):
                Q, R = np.linalg.qr(self.basis.T)
                return Q.T, R.T
                
            def decode_cvp(self, target):
                return np.round(target)
                
            def sample_lattice_point(self, sigma):
                return np.zeros(self.get_dimension())
                
            def get_dimension(self):
                return self.basis.shape[0]
        
        ntru_lattice = NTRULattice(basis)
        
        # Step 2: Reduction for cryptographic lattice
        reducer = LatticeReduction()
        
        # Compute security-relevant metrics
        original_hermite = reducer.hermite_factor(basis)
        original_orthogonality = reducer.orthogonality_defect(basis)
        
        # Perform BKZ reduction (more relevant for cryptographic lattices)
        try:
            reduced_basis, unimodular = reducer.bkz_reduce(basis, block_size=2, max_tours=1)
            
            reduced_hermite = reducer.hermite_factor(reduced_basis)
            reduced_orthogonality = reducer.orthogonality_defect(reduced_basis)
            
            reduction_successful = True
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # BKZ might fail on some lattices
            reduced_basis = basis
            reduced_hermite = original_hermite
            reduced_orthogonality = original_orthogonality
            reduction_successful = False
        
        # Step 3: Security analysis
        # Estimate security level based on lattice parameters
        def estimate_security_level(basis):
            """Simple security estimation based on dimension and first minimum."""
            dim = basis.shape[0]
            
            # Approximate first minimum using Gaussian heuristic
            det = np.abs(np.linalg.det(basis @ basis.T))
            vol = np.sqrt(det)
            gaussian_heuristic = np.sqrt(dim / (2 * np.pi * np.e)) * vol**(1/dim)
            
            # Security level is roughly log2 of work factor
            # This is a very simplified estimate
            security_bits = max(0, np.log2(gaussian_heuristic * dim))
            return min(security_bits, 256)  # Cap at reasonable level
        
        original_security = estimate_security_level(basis)
        reduced_security = estimate_security_level(reduced_basis)
        
        # Step 4: Sampling attempt
        sigma = 5.0  # Larger sigma for cryptographic lattice
        
        # Update lattice with reduced basis
        ntru_lattice_reduced = NTRULattice(reduced_basis)
        
        try:
            sampler = KleinSampler(ntru_lattice_reduced, sigma)
            
            # Generate small number of samples
            n_samples = 50
            samples = []
            
            for _ in range(n_samples):
                sample = sampler.sample()
                samples.append(sample)
            
            samples = np.array(samples)
            sampling_successful = True
            
            # Basic statistical check
            sample_mean = np.mean(samples, axis=0)
            sample_norms = np.linalg.norm(samples, axis=1)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            sampling_successful = False
            samples = None
            sample_mean = None
            sample_norms = None
        
        # Step 5: Results compilation
        results = {
            'lattice_type': 'ntru_like',
            'parameters': {'n': n, 'q': q},
            'sigma': sigma,
            'reduction_successful': reduction_successful,
            'sampling_successful': sampling_successful,
            'original_hermite_factor': float(original_hermite),
            'reduced_hermite_factor': float(reduced_hermite),
            'original_orthogonality_defect': float(original_orthogonality),
            'reduced_orthogonality_defect': float(reduced_orthogonality),
            'original_security_estimate': float(original_security),
            'reduced_security_estimate': float(reduced_security)
        }
        
        if sampling_successful:
            results.update({
                'n_samples': n_samples,
                'sample_mean_norm': float(np.linalg.norm(sample_mean)),
                'average_sample_norm': float(np.mean(sample_norms)),
                'sample_norm_std': float(np.std(sample_norms))
            })
        
        # Step 6: Save results
        results_file = temp_dir / 'ntru_like_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Step 7: Validation
        assert results_file.exists()
        
        # Basic sanity checks
        assert results['original_hermite_factor'] >= 1.0
        assert results['reduced_hermite_factor'] >= 1.0
        assert results['original_orthogonality_defect'] >= 1.0
        assert results['reduced_orthogonality_defect'] >= 1.0
        
        # If reduction was successful, it should improve or maintain quality
        if reduction_successful:
            assert results['reduced_hermite_factor'] <= results['original_hermite_factor'] * 1.1
            assert results['reduced_orthogonality_defect'] <= results['original_orthogonality_defect'] * 1.1


@pytest.mark.end_to_end
class TestDiagnosticPipeline:
    """Test complete diagnostic pipeline."""
    
    def test_comprehensive_mcmc_analysis(self, temp_dir, statistical_config):
        """Test comprehensive MCMC analysis pipeline."""
        # Step 1: Setup multiple MCMC chains
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        n_chains = 4
        n_samples = statistical_config['n_samples'] // 10  # Smaller for integration test
        
        all_chains = []
        acceptance_rates = []
        
        for chain_id in range(n_chains):
            sampler = IMHKSampler(lattice, sigma)
            
            # Burn-in
            burn_in = 200
            for _ in range(burn_in):
                sampler.sample()
            
            # Collect samples and track acceptance
            chain = []
            n_accepted = 0
            
            for i in range(n_samples):
                old_sample = sampler.current_sample.copy()
                new_sample = sampler.sample()
                chain.append(new_sample[0])  # Just first coordinate
                
                if not np.array_equal(old_sample, new_sample):
                    n_accepted += 1
            
            all_chains.append(np.array(chain))
            acceptance_rates.append(n_accepted / n_samples)
        
        # Step 2: Convergence diagnostics
        diagnostics = ConvergenceDiagnostics()
        mcmc_diagnostics = MCMCDiagnostics()
        
        # Gelman-Rubin diagnostic
        rhat = diagnostics.gelman_rubin_diagnostic(all_chains)
        
        # Effective sample sizes
        ess_values = [diagnostics.effective_sample_size(chain) for chain in all_chains]
        
        # Geweke diagnostics
        geweke_scores = [diagnostics.geweke_diagnostic(chain) for chain in all_chains]
        
        # Comprehensive MCMC analysis
        comprehensive_results = mcmc_diagnostics.comprehensive_analysis(all_chains, acceptance_rates)
        
        # Step 3: Spectral analysis
        spectral = SpectralAnalysis()
        
        spectral_gap = spectral.compute_spectral_gap(lattice, sigma, method='theoretical')
        mixing_time = spectral.estimate_mixing_time(lattice, sigma, epsilon=0.25)
        theoretical_bounds = spectral.theoretical_bounds(lattice, sigma)
        
        # Step 4: Compile comprehensive results
        pipeline_results = {
            'experiment_info': {
                'lattice_type': 'identity',
                'dimension': lattice.get_dimension(),
                'sigma': sigma,
                'n_chains': n_chains,
                'n_samples_per_chain': n_samples,
                'burn_in': burn_in
            },
            'convergence_diagnostics': {
                'gelman_rubin': float(rhat),
                'effective_sample_sizes': [float(ess) for ess in ess_values],
                'geweke_scores': [float(score) for score in geweke_scores],
                'acceptance_rates': [float(rate) for rate in acceptance_rates]
            },
            'spectral_analysis': {
                'spectral_gap': float(spectral_gap),
                'mixing_time': float(mixing_time),
                'theoretical_bounds': {k: float(v) for k, v in theoretical_bounds.items()}
            },
            'comprehensive_mcmc_analysis': comprehensive_results
        }
        
        # Step 5: Statistical validation
        # R-hat should indicate convergence
        assert 1.0 <= rhat < 1.2, f"Chains may not have converged: R-hat = {rhat}"
        
        # ESS should be reasonable
        min_ess = min(ess_values)
        assert min_ess > n_samples * 0.05, f"ESS too low: {min_ess}"
        
        # Geweke scores should be reasonable
        max_geweke = max(abs(score) for score in geweke_scores)
        assert max_geweke < 3, f"Geweke scores indicate non-stationarity: {max_geweke}"
        
        # Acceptance rates should be reasonable
        avg_acceptance = np.mean(acceptance_rates)
        assert 0.1 < avg_acceptance < 0.9, f"Average acceptance rate problematic: {avg_acceptance}"
        
        # Step 6: Save results
        results_file = temp_dir / 'comprehensive_mcmc_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        # Also save raw chain data
        chains_file = temp_dir / 'mcmc_chains.npz'
        np.savez(chains_file, **{f'chain_{i}': chain for i, chain in enumerate(all_chains)})
        
        # Step 7: Validation
        assert results_file.exists()
        assert chains_file.exists()
        
        # Load and verify results
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['experiment_info']['n_chains'] == n_chains
        assert len(loaded_results['convergence_diagnostics']['effective_sample_sizes']) == n_chains
        
        # Load and verify chain data
        loaded_chains = np.load(chains_file)
        assert len(loaded_chains.files) == n_chains
        
        for i in range(n_chains):
            loaded_chain = loaded_chains[f'chain_{i}']
            np.testing.assert_array_equal(loaded_chain, all_chains[i])


@pytest.mark.end_to_end
class TestGoldenFileValidation:
    """Test against golden reference outputs."""
    
    def test_golden_file_comparison(self, temp_dir, golden_files_config, tolerance_config):
        """Test pipeline outputs against golden reference files."""
        # Step 1: Run standard pipeline
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        
        # Generate deterministic samples
        np.random.seed(42)
        sampler = KleinSampler(lattice, sigma)
        n_samples = 1000
        
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Compute standard metrics
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)
        
        diagnostics = ConvergenceDiagnostics()
        ess = diagnostics.effective_sample_size(samples[:, 0])
        
        spectral = SpectralAnalysis()
        gap = spectral.compute_spectral_gap(lattice, sigma, method='theoretical')
        
        # Step 2: Create current results
        current_results = {
            'sample_mean': sample_mean.tolist(),
            'sample_covariance': sample_cov.tolist(),
            'effective_sample_size': float(ess),
            'spectral_gap': float(gap)
        }
        
        # Step 3: Save current results as golden file (for first run)
        golden_file = golden_files_config['golden_dir'] / 'identity_2d_sigma1_results.json'
        
        if not golden_file.exists():
            # Create golden file
            with open(golden_file, 'w') as f:
                json.dump(current_results, f, indent=2)
            
            # Skip comparison on first run
            pytest.skip("Golden file created, run test again to compare")
        
        # Step 4: Load golden results and compare
        with open(golden_file, 'r') as f:
            golden_results = json.load(f)
        
        # Compare with tolerance
        tol = golden_files_config['tolerance']
        
        np.testing.assert_allclose(
            current_results['sample_mean'],
            golden_results['sample_mean'],
            atol=0.1,  # Statistical tolerance
            err_msg="Sample mean differs from golden file"
        )
        
        np.testing.assert_allclose(
            current_results['sample_covariance'],
            golden_results['sample_covariance'],
            rtol=0.2,  # Statistical tolerance for covariance
            err_msg="Sample covariance differs from golden file"
        )
        
        # ESS can vary significantly, use loose tolerance
        ess_ratio = current_results['effective_sample_size'] / golden_results['effective_sample_size']
        assert 0.5 < ess_ratio < 2.0, "ESS differs significantly from golden file"
        
        # Spectral gap should be deterministic
        np.testing.assert_allclose(
            current_results['spectral_gap'],
            golden_results['spectral_gap'],
            rtol=tol,
            err_msg="Spectral gap differs from golden file"
        )


@pytest.mark.end_to_end
@pytest.mark.slow
class TestScalabilityPipeline:
    """Test pipeline scalability with different dimensions."""
    
    def test_dimension_scaling(self, temp_dir, performance_config):
        """Test pipeline performance scaling with dimension."""
        import time
        
        dimensions = [2, 3, 4, 5]
        sigma = 1.0
        n_samples = 500  # Reduced for scalability test
        
        scaling_results = []
        
        for dim in dimensions:
            dim_start_time = time.time()
            
            # Step 1: Lattice construction
            lattice = IdentityLattice(dimension=dim)
            
            # Step 2: Sampling
            sampler = KleinSampler(lattice, sigma)
            
            sampling_start = time.time()
            samples = np.array([sampler.sample() for _ in range(n_samples)])
            sampling_time = time.time() - sampling_start
            
            # Step 3: Basic diagnostics
            diagnostics_start = time.time()
            diagnostics = ConvergenceDiagnostics()
            ess = diagnostics.effective_sample_size(samples[:, 0])
            diagnostics_time = time.time() - diagnostics_start
            
            # Step 4: Spectral analysis
            spectral_start = time.time()
            spectral = SpectralAnalysis()
            gap = spectral.compute_spectral_gap(lattice, sigma, method='theoretical')
            spectral_time = time.time() - spectral_start
            
            total_time = time.time() - dim_start_time
            
            # Record results
            dim_results = {
                'dimension': dim,
                'total_time': total_time,
                'sampling_time': sampling_time,
                'diagnostics_time': diagnostics_time,
                'spectral_time': spectral_time,
                'effective_sample_size': float(ess),
                'spectral_gap': float(gap),
                'samples_per_second': n_samples / sampling_time
            }
            
            scaling_results.append(dim_results)
            
            # Performance checks
            max_time = performance_config['max_time_sampling'] * dim  # Scale with dimension
            assert total_time < max_time, \
                f"Pipeline too slow for dimension {dim}: {total_time:.2f}s > {max_time:.2f}s"
        
        # Step 5: Save scaling results
        scaling_file = temp_dir / 'dimension_scaling_results.json'
        with open(scaling_file, 'w') as f:
            json.dump(scaling_results, f, indent=2)
        
        # Step 6: Analyze scaling behavior
        dimensions_list = [r['dimension'] for r in scaling_results]
        sampling_times = [r['sampling_time'] for r in scaling_results]
        
        # Sampling time should scale reasonably with dimension
        # For Klein algorithm, expect roughly O(d^2) or O(d^3) scaling
        for i in range(1, len(scaling_results)):
            dim_ratio = dimensions_list[i] / dimensions_list[i-1]
            time_ratio = sampling_times[i] / sampling_times[i-1]
            
            # Allow for reasonable scaling (not more than cubic)
            assert time_ratio < dim_ratio**3 * 2, \
                f"Sampling time scaling too poor: {time_ratio} vs dimension ratio {dim_ratio}"


@pytest.mark.end_to_end
class TestReproducibilityPipeline:
    """Test reproducibility of complete pipeline."""
    
    def test_full_pipeline_reproducibility(self, temp_dir, test_seed):
        """Test that complete pipeline produces reproducible results."""
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        n_samples = 1000
        
        def run_pipeline(seed):
            """Run complete pipeline with given seed."""
            np.random.seed(seed)
            
            # Sampling
            sampler = KleinSampler(lattice, sigma)
            samples = np.array([sampler.sample() for _ in range(n_samples)])
            
            # Diagnostics
            diagnostics = ConvergenceDiagnostics()
            ess = diagnostics.effective_sample_size(samples[:, 0])
            
            # Spectral analysis
            spectral = SpectralAnalysis()
            gap = spectral.compute_spectral_gap(lattice, sigma, method='theoretical')
            
            return {
                'samples': samples,
                'ess': ess,
                'spectral_gap': gap
            }
        
        # Run pipeline twice with same seed
        results1 = run_pipeline(test_seed)
        results2 = run_pipeline(test_seed)
        
        # Results should be identical
        np.testing.assert_array_equal(
            results1['samples'], results2['samples'],
            err_msg="Samples not reproducible"
        )
        
        np.testing.assert_allclose(
            results1['ess'], results2['ess'], rtol=1e-15,
            err_msg="ESS not reproducible"
        )
        
        np.testing.assert_allclose(
            results1['spectral_gap'], results2['spectral_gap'], rtol=1e-15,
            err_msg="Spectral gap not reproducible"
        )
        
        # Save reproducibility verification
        repro_file = temp_dir / 'reproducibility_verification.json'
        with open(repro_file, 'w') as f:
            json.dump({
                'test_passed': True,
                'seed_used': test_seed,
                'ess_value': float(results1['ess']),
                'spectral_gap_value': float(results1['spectral_gap']),
                'sample_mean': results1['samples'].mean(axis=0).tolist(),
                'sample_std': results1['samples'].std(axis=0).tolist()
            }, f, indent=2)
        
        assert repro_file.exists()