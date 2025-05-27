"""
Comprehensive experimental study of lattice Gaussian sampling for cryptographic applications.

Focuses on convergence diagnostics, sample quality, mixing time, and parameter sensitivity
for Falcon/NTRU applications. Implements and analyzes Klein and IMHK (MH-Klein) algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, pi, sqrt, exp, log, ln,
    identity_matrix, random_matrix, GF, next_prime, jacobi_theta_3
)
from multiprocessing import Pool, cpu_count
import pickle
import os
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from scipy import stats
from collections import defaultdict
import time
import traceback

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
})

# Import our implementations
import sys
sys.path.append('..')
from src.lattices.base import Lattice
from src.lattices.ntru import NTRULattice
from src.lattices.qary import QaryLattice
from src.lattices.identity import IdentityLattice
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IndependentMHK
from src.diagnostics.spectral import SpectralGapAnalyzer
from src.diagnostics.convergence import ConvergenceDiagnostics


class CryptographicLatticeExperiments:
    """
    Comprehensive experimental framework for cryptographic lattice Gaussian sampling.
    """
    
    def __init__(self, output_dir='../results/cryptographic', n_cores=None, random_seed=42):
        """
        Initialize experimental framework.
        
        Args:
            output_dir: Directory for all outputs
            n_cores: Number of CPU cores for parallelization
            random_seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.n_cores = n_cores or cpu_count()
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        from sage.all import set_random_seed
        set_random_seed(random_seed)
        
        # Create output directories
        self.dirs = {}
        for subdir in ['figures', 'tables', 'logs', 'checkpoints', 'data']:
            path = os.path.join(output_dir, subdir)
            os.makedirs(path, exist_ok=True)
            self.dirs[subdir] = path
            
        # Experiment metadata
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metadata = {
            'experiment_id': self.experiment_id,
            'random_seed': random_seed,
            'n_cores': self.n_cores,
            'start_time': datetime.now().isoformat()
        }
        
        # Log setup
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging infrastructure."""
        import logging
        
        log_file = os.path.join(self.dirs['logs'], f'experiment_{self.experiment_id}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized experiment {self.experiment_id}")
        
    # ========== 1. Lattice Suite and Parameters ==========
    
    def generate_lattice_suite(self):
        """
        Generate comprehensive lattice test suite for cryptographic applications.
        
        Returns:
            dict: Lattices organized by type and parameters
        """
        self.logger.info("Generating lattice test suite...")
        lattices = {}
        
        # Checkerboard lattices D_n
        for n in [4, 8, 16, 32, 64]:
            lattices[('checkerboard', n)] = self._create_checkerboard_lattice(n)
            self.logger.info(f"Created checkerboard lattice D_{n}")
            
        # Random q-ary lattices with varying conditioning
        q_params = [
            (16, 127, 'well-conditioned'),
            (32, 3329, 'moderate'),
            (64, 12289, 'poor-conditioned')
        ]
        
        for n, q, condition in q_params:
            lattice = self._create_qary_lattice(n, q, condition)
            lattices[('qary', n, condition)] = lattice
            self.logger.info(f"Created q-ary lattice: n={n}, q={q}, {condition}")
            
        # NTRU lattices with cryptographic parameters
        ntru_params = [
            (256, 7681, 1.17, 'Falcon-256'),   # Falcon-256 parameters
            (512, 12289, 1.17, 'Falcon-512'),  # Falcon-512 parameters
            (1024, 12289, 1.17, 'Falcon-1024') # Falcon-1024 parameters
        ]
        
        for n, q, sigma_factor, name in ntru_params:
            lattice = self._create_ntru_lattice(n, q, sigma_factor)
            lattices[('ntru', n, name)] = lattice
            self.logger.info(f"Created NTRU lattice: {name}")
            
        # Save lattice metadata
        self._save_lattice_metadata(lattices)
        
        return lattices
    
    def _create_checkerboard_lattice(self, n):
        """Create n-dimensional checkerboard lattice D_n."""
        return CheckerboardLattice(n)
    
    def _create_qary_lattice(self, n, q, condition_type):
        """Create q-ary lattice with specified conditioning."""
        if condition_type == 'well-conditioned':
            # Use identity + small perturbation
            A = identity_matrix(ZZ, n)
            A = A + random_matrix(ZZ, n, n, x=-1, y=2)
        elif condition_type == 'moderate':
            # Random matrix mod q
            A = random_matrix(ZZ, n, n, x=0, y=q)
        else:  # poor-conditioned
            # Sparse matrix for poor conditioning
            A = Matrix(ZZ, n, n)
            for i in range(n):
                A[i, i] = q - 1
                if i < n - 1:
                    A[i, i+1] = 1
                    
        return QaryLattice(n, q, A)
    
    def _create_ntru_lattice(self, n, q, sigma_factor):
        """Create NTRU lattice with Falcon-like parameters."""
        # Standard deviation for polynomial sampling
        sigma = sigma_factor * sqrt(q / (2 * n))
        return NTRULattice(n, q, float(sigma))
    
    # ========== 2. Convergence and Mixing Experiments ==========
    
    def compare_algorithms(self, lattices, sigma_values=None, n_samples=10000, n_chains=5):
        """
        Compare Klein and MH-Klein algorithms on all test lattices.
        
        Args:
            lattices: Dictionary of test lattices
            sigma_values: List of σ values to test (auto if None)
            n_samples: Samples per chain
            n_chains: Number of independent chains
            
        Returns:
            dict: Comprehensive comparison results
        """
        self.logger.info("Starting algorithm comparison experiments...")
        results = defaultdict(dict)
        
        for lattice_key, lattice in lattices.items():
            self.logger.info(f"Testing lattice: {lattice_key}")
            
            # Determine σ values if not provided
            if sigma_values is None:
                sigma_values = self._determine_sigma_range(lattice)
                
            for sigma in sigma_values:
                self.logger.info(f"  Testing σ={sigma:.4f}")
                
                # Run Klein algorithm
                klein_results = self._run_klein_experiment(
                    lattice, sigma, n_samples, n_chains
                )
                
                # Run MH-Klein (IMHK) algorithm
                mhk_results = self._run_mhk_experiment(
                    lattice, sigma, n_samples, n_chains
                )
                
                # Store results
                results[lattice_key][sigma] = {
                    'klein': klein_results,
                    'mhk': mhk_results,
                    'comparison': self._compare_results(klein_results, mhk_results)
                }
                
                # Save checkpoint
                self._save_checkpoint(results, 'algorithm_comparison')
                
        # Generate comparison plots
        self._plot_algorithm_comparison(results)
        
        return results
    
    def _run_klein_experiment(self, lattice, sigma, n_samples, n_chains):
        """Run Klein algorithm experiment."""
        start_time = time.time()
        
        # Create sampler
        sampler = KleinSampler(lattice, sigma)
        
        # Run multiple chains in parallel
        with Pool(self.n_cores) as pool:
            chains = pool.starmap(
                self._run_single_klein_chain,
                [(lattice, sigma, n_samples, i) for i in range(n_chains)]
            )
            
        # Analyze results
        results = {
            'chains': chains,
            'time_elapsed': time.time() - start_time,
            'samples_per_second': (n_samples * n_chains) / (time.time() - start_time)
        }
        
        # Compute diagnostics
        diagnostics = ConvergenceDiagnostics(lattice, sigma)
        pooled_samples = np.vstack(chains)
        
        results['diagnostics'] = {
            'effective_sample_size': diagnostics.effective_sample_size(pooled_samples),
            'distance_to_mode': diagnostics.distance_to_mode(
                pooled_samples, lattice, sampler.center
            )
        }
        
        return results
    
    def _run_mhk_experiment(self, lattice, sigma, n_samples, n_chains):
        """Run MH-Klein experiment with full diagnostics."""
        start_time = time.time()
        
        # Create sampler
        sampler = IndependentMHK(lattice, sigma)
        
        # Theoretical predictions
        theoretical = {
            'delta': sampler.compute_delta(),
            'spectral_gap': sampler.spectral_gap(),
            'mixing_time': sampler.mixing_time()
        }
        
        # Run chains
        chains = []
        acceptance_rates = []
        tvd_history = []
        
        for i in range(n_chains):
            self.logger.info(f"    Running chain {i+1}/{n_chains}")
            
            # Full chain with TVD tracking
            chain, tvd_trace = self._run_mhk_chain_with_diagnostics(
                sampler, n_samples
            )
            
            chains.append(chain)
            acceptance_rates.append(sampler.get_acceptance_rate())
            tvd_history.append(tvd_trace)
            
            # Reset sampler for next chain
            sampler.n_accepted = 0
            sampler.n_proposed = 0
            
        # Analyze results
        results = {
            'chains': chains,
            'time_elapsed': time.time() - start_time,
            'theoretical': theoretical,
            'acceptance_rates': acceptance_rates,
            'tvd_history': tvd_history
        }
        
        # Convergence diagnostics
        diagnostics = ConvergenceDiagnostics(lattice, sigma)
        results['empirical_mixing_time'] = diagnostics.empirical_mixing_time(chains)
        
        # Spectral gap analysis
        spectral_analyzer = SpectralGapAnalyzer(sampler)
        pooled = np.vstack([c[results['empirical_mixing_time']:] for c in chains])
        
        if len(pooled) > 1000:  # Only for reasonable sample sizes
            results['empirical_spectral_gap'] = spectral_analyzer.empirical_spectral_gap(
                pooled[:1000]  # Use subset for efficiency
            )
            
        return results
    
    def _run_mhk_chain_with_diagnostics(self, sampler, n_samples):
        """Run MHK chain with TVD tracking."""
        chain = []
        tvd_trace = []
        
        # Initialize with Klein sample
        sampler.current_state = sampler.klein_sampler.sample()
        
        # Checkpoints for TVD computation
        checkpoints = np.unique(np.logspace(0, np.log10(n_samples), 50, dtype=int))
        checkpoint_idx = 0
        
        for t in range(n_samples):
            sampler._mh_step()
            chain.append(np.array(sampler.current_state))
            
            # Compute TVD at checkpoints
            if checkpoint_idx < len(checkpoints) and t == checkpoints[checkpoint_idx]:
                if len(chain) > 100:  # Need enough samples
                    tvd = self._estimate_tvd_to_klein(
                        np.array(chain[-100:]), sampler
                    )
                    tvd_trace.append((t, tvd))
                checkpoint_idx += 1
                
        return np.array(chain), tvd_trace
    
    def _estimate_tvd_to_klein(self, mhk_samples, sampler):
        """Estimate TVD between MHK samples and Klein distribution."""
        # Generate Klein reference samples
        klein = KleinSampler(sampler.lattice_sage, float(sampler.sigma_sage))
        klein_samples = np.array([klein.sample() for _ in range(len(mhk_samples))])
        
        # Simple TVD estimation using histogram
        n_bins = int(np.sqrt(len(mhk_samples)))
        
        # Use first component for simplicity
        hist_mhk, bins = np.histogram(mhk_samples[:, 0], bins=n_bins, density=True)
        hist_klein, _ = np.histogram(klein_samples[:, 0], bins=bins, density=True)
        
        # Normalize
        hist_mhk = hist_mhk / hist_mhk.sum()
        hist_klein = hist_klein / hist_klein.sum()
        
        # TVD
        tvd = 0.5 * np.sum(np.abs(hist_mhk - hist_klein))
        
        return tvd
    
    # ========== 3. Dimension and Parameter Scaling ==========
    
    def dimension_scaling_experiment(self, dimensions=[10, 20, 50, 100, 200, 500, 1000],
                                   sigma_ratio=2.0, n_samples=5000):
        """
        Study how performance scales with lattice dimension.
        
        Args:
            dimensions: List of dimensions to test
            sigma_ratio: σ/||b*_max|| ratio for fair comparison
            n_samples: Samples per test
            
        Returns:
            dict: Scaling analysis results
        """
        self.logger.info("Starting dimension scaling experiments...")
        results = {
            'dimensions': dimensions,
            'klein_times': [],
            'mhk_times': [],
            'theoretical_mixing': [],
            'empirical_mixing': [],
            'spectral_gaps': [],
            'memory_usage': []
        }
        
        for n in dimensions:
            self.logger.info(f"Testing dimension n={n}")
            
            try:
                # Create checkerboard lattice
                lattice = CheckerboardLattice(n)
                
                # Set σ based on Gram-Schmidt norms
                gs_norms = lattice.get_gram_schmidt_norms()
                sigma = sigma_ratio * max(gs_norms)
                
                # Time Klein algorithm
                klein_time = self._time_klein_sampling(lattice, sigma, n_samples)
                results['klein_times'].append(klein_time)
                
                # Analyze MHK
                mhk_analysis = self._analyze_mhk_scaling(lattice, sigma, n_samples)
                results['mhk_times'].append(mhk_analysis['time'])
                results['theoretical_mixing'].append(mhk_analysis['theoretical_mixing'])
                results['empirical_mixing'].append(mhk_analysis['empirical_mixing'])
                results['spectral_gaps'].append(mhk_analysis['spectral_gap'])
                
                # Memory usage
                import psutil
                process = psutil.Process()
                results['memory_usage'].append(process.memory_info().rss / 1024**2)  # MB
                
                # Save checkpoint
                self._save_checkpoint(results, f'dimension_scaling_n{n}')
                
            except Exception as e:
                self.logger.error(f"Error at dimension {n}: {str(e)}")
                self.logger.error(traceback.format_exc())
                # Append None for failed experiments
                for key in ['klein_times', 'mhk_times', 'theoretical_mixing', 
                           'empirical_mixing', 'spectral_gaps', 'memory_usage']:
                    if len(results[key]) < len(results['dimensions'][:results['dimensions'].index(n)+1]):
                        results[key].append(None)
                        
        # Generate scaling plots
        self._plot_dimension_scaling(results)
        
        # Save results table
        self._save_scaling_table(results)
        
        return results
    
    # ========== 4. Parameter Sensitivity and Robustness ==========
    
    def sigma_sensitivity(self, lattice, n_samples=10000):
        """
        Analyze sensitivity to σ parameter for cryptographic applications.
        
        Args:
            lattice: Test lattice
            n_samples: Samples per σ value
            
        Returns:
            dict: Sensitivity analysis results
        """
        self.logger.info("Starting σ sensitivity analysis...")
        
        # Determine σ range
        smoothing = lattice.smoothing_parameter()
        gs_norms = lattice.get_gram_schmidt_norms()
        klein_min = sqrt(log(lattice.get_dimension())) * max(gs_norms)
        
        # Falcon practical range
        falcon_min = 1.17 * sqrt(lattice.get_dimension())
        falcon_max = 1.5 * sqrt(lattice.get_dimension())
        
        # Test range
        sigma_values = np.unique(np.concatenate([
            np.linspace(smoothing, klein_min * 0.8, 10),
            np.linspace(klein_min * 0.8, klein_min * 1.2, 10),
            np.linspace(falcon_min, falcon_max, 5)
        ]))
        
        results = {
            'sigma': [],
            'mixing_time': [],
            'acceptance_rate': [],
            'tvd_to_target': [],
            'chi_squared': [],
            'effective_sample_size': [],
            'recommendation': []
        }
        
        for sigma in sigma_values:
            self.logger.info(f"Testing σ={sigma:.4f}")
            
            # Run experiments
            analysis = self._analyze_sigma_choice(lattice, sigma, n_samples)
            
            # Store results
            results['sigma'].append(sigma)
            for key in ['mixing_time', 'acceptance_rate', 'tvd_to_target',
                       'chi_squared', 'effective_sample_size']:
                results[key].append(analysis[key])
                
            # Recommendation
            if sigma < smoothing:
                rec = 'too_small'
            elif sigma < klein_min * 0.9:
                rec = 'marginal'
            elif falcon_min <= sigma <= falcon_max:
                rec = 'optimal_crypto'
            else:
                rec = 'safe'
            results['recommendation'].append(rec)
            
        # Identify optimal σ ranges
        self._analyze_optimal_sigma_ranges(results)
        
        # Generate sensitivity plots
        self._plot_sigma_sensitivity(results)
        
        return results
    
    # ========== 5. Empirical Validation ==========
    
    def validate_spectral_gap(self, samplers_config, n_samples=50000):
        """
        Compare empirical and theoretical spectral gaps.
        
        Args:
            samplers_config: List of (lattice, sigma) pairs
            n_samples: Samples for empirical estimation
            
        Returns:
            dict: Validation results
        """
        self.logger.info("Validating spectral gap predictions...")
        results = []
        
        for lattice, sigma in samplers_config:
            self.logger.info(f"Testing lattice dim={lattice.get_dimension()}, σ={sigma:.4f}")
            
            # Create sampler
            sampler = IndependentMHK(lattice, sigma)
            
            # Theoretical prediction
            theoretical_gap = sampler.spectral_gap()
            
            # Generate samples
            samples = sampler.sample(n_samples)
            
            # Empirical estimation
            analyzer = SpectralGapAnalyzer(sampler)
            empirical_gap = analyzer.empirical_spectral_gap(samples[:5000])  # Use subset
            
            # Confidence interval via bootstrap
            gaps = []
            for _ in range(100):
                idx = np.random.choice(5000, 5000, replace=True)
                gap = analyzer.empirical_spectral_gap(samples[idx])
                gaps.append(gap)
                
            ci_lower = np.percentile(gaps, 2.5)
            ci_upper = np.percentile(gaps, 97.5)
            
            results.append({
                'lattice_dim': lattice.get_dimension(),
                'sigma': sigma,
                'theoretical_gap': theoretical_gap,
                'empirical_gap': empirical_gap,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'relative_error': abs(empirical_gap - theoretical_gap) / theoretical_gap
            })
            
        # Create validation plot
        self._plot_spectral_gap_validation(results)
        
        return results
    
    def validate_tvd_bound(self, sampler, n_chains=10, chain_length=10000):
        """
        Validate theoretical TVD decay bounds.
        
        Args:
            sampler: MHK sampler
            n_chains: Number of independent chains
            chain_length: Length of each chain
            
        Returns:
            dict: TVD validation results
        """
        self.logger.info("Validating TVD bounds...")
        
        # Run chains
        chains = sampler.parallel_chains(n_chains, chain_length)
        
        # Checkpoints
        checkpoints = np.unique(np.logspace(1, np.log10(chain_length), 30, dtype=int))
        
        empirical_tvd = []
        empirical_ci = []
        theoretical_tvd = []
        
        for t in checkpoints:
            # Empirical TVD estimation
            tvds = []
            for chain in chains:
                if t < len(chain):
                    tvd = self._estimate_tvd_to_target(chain[:t], sampler)
                    tvds.append(tvd)
                    
            if tvds:
                empirical_tvd.append(np.mean(tvds))
                empirical_ci.append((np.percentile(tvds, 2.5), np.percentile(tvds, 97.5)))
            else:
                empirical_tvd.append(None)
                empirical_ci.append((None, None))
                
            # Theoretical bound
            theoretical_tvd.append(sampler.total_variation_distance(t))
            
        # Plot validation
        self._plot_tvd_validation(checkpoints, empirical_tvd, empirical_ci, theoretical_tvd)
        
        return {
            'checkpoints': checkpoints,
            'empirical_tvd': empirical_tvd,
            'empirical_ci': empirical_ci,
            'theoretical_tvd': theoretical_tvd
        }
    
    # ========== 6. Publication-Quality Outputs ==========
    
    def generate_all_outputs(self, results):
        """
        Generate all publication-quality outputs.
        
        Args:
            results: Dictionary of all experimental results
        """
        self.logger.info("Generating publication outputs...")
        
        # LaTeX tables
        self._generate_latex_tables(results)
        
        # Summary statistics
        self._generate_summary_statistics(results)
        
        # README
        self._generate_readme()
        
        # Final metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_runtime'] = str(
            datetime.now() - datetime.fromisoformat(self.metadata['start_time'])
        )
        
        # Save metadata
        with open(os.path.join(self.dirs['logs'], 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        self.logger.info("All outputs generated successfully!")
        
    # ========== Helper Methods ==========
    
    def _determine_sigma_range(self, lattice):
        """Determine appropriate σ values for a lattice."""
        gs_norms = lattice.get_gram_schmidt_norms()
        max_norm = max(gs_norms)
        
        # Range from near-smoothing to well above Klein's requirement
        return np.logspace(
            np.log10(max_norm),
            np.log10(10 * max_norm),
            10
        )
        
    def _run_single_klein_chain(self, lattice, sigma, n_samples, chain_id):
        """Run a single Klein chain (for multiprocessing)."""
        np.random.seed(self.random_seed + chain_id)
        sampler = KleinSampler(lattice, sigma)
        
        samples = []
        for _ in range(n_samples):
            samples.append(sampler.sample())
            
        return np.array(samples)
    
    def _compare_results(self, klein_results, mhk_results):
        """Compare Klein and MHK results."""
        comparison = {
            'speedup': klein_results['samples_per_second'] / (
                len(mhk_results['chains'][0]) * len(mhk_results['chains']) / 
                mhk_results['time_elapsed']
            ),
            'mixing_time_ratio': (
                mhk_results['empirical_mixing_time'] / 1.0 
                if 'empirical_mixing_time' in mhk_results else None
            ),
            'acceptance_rate': np.mean(mhk_results['acceptance_rates'])
        }
        
        return comparison
    
    def _save_checkpoint(self, data, name):
        """Save checkpoint data."""
        checkpoint_file = os.path.join(
            self.dirs['checkpoints'],
            f'{name}_{self.experiment_id}.pkl'
        )
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
            
    def _save_lattice_metadata(self, lattices):
        """Save lattice metadata."""
        metadata = {}
        
        for key, lattice in lattices.items():
            metadata[str(key)] = {
                'dimension': lattice.get_dimension(),
                'determinant': float(lattice.get_determinant()),
                'max_gs_norm': float(max(lattice.get_gram_schmidt_norms())),
                'smoothing_parameter': float(lattice.smoothing_parameter())
            }
            
        with open(os.path.join(self.dirs['data'], 'lattice_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _plot_algorithm_comparison(self, results):
        """Create algorithm comparison plots."""
        # Plot 1: Mixing time comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data for plotting
        for i, (lattice_key, sigma_results) in enumerate(results.items()):
            if i >= 4:
                break
                
            ax = axes[i // 2, i % 2]
            
            sigmas = sorted(sigma_results.keys())
            klein_times = []
            mhk_mixing = []
            mhk_acceptance = []
            
            for sigma in sigmas:
                data = sigma_results[sigma]
                klein_times.append(data['klein']['samples_per_second'])
                
                if 'empirical_mixing_time' in data['mhk']:
                    mhk_mixing.append(data['mhk']['empirical_mixing_time'])
                else:
                    mhk_mixing.append(None)
                    
                mhk_acceptance.append(np.mean(data['mhk']['acceptance_rates']))
                
            # Plot
            ax2 = ax.twinx()
            
            line1 = ax.plot(sigmas, klein_times, 'b-', label='Klein (samples/s)')
            line2 = ax.plot(sigmas, mhk_mixing, 'r--', label='MHK mixing time')
            line3 = ax2.plot(sigmas, mhk_acceptance, 'g:', label='MHK acceptance')
            
            ax.set_xlabel(r'$\sigma$')
            ax.set_ylabel('Klein samples/s | MHK mixing time')
            ax2.set_ylabel('MHK acceptance rate')
            ax.set_title(f'Lattice: {lattice_key}')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'algorithm_comparison.pdf'))
        plt.close()
        
        # Plot 2: TVD decay for MHK
        self._plot_tvd_decay_comparison(results)
        
    def _plot_tvd_decay_comparison(self, results):
        """Plot TVD decay for different configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        plot_idx = 0
        for lattice_key, sigma_results in results.items():
            if plot_idx >= 4:
                break
                
            for sigma, data in sigma_results.items():
                if 'tvd_history' in data['mhk'] and plot_idx < 4:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    
                    # Plot all chains
                    for chain_tvd in data['mhk']['tvd_history']:
                        if chain_tvd:
                            times, tvds = zip(*chain_tvd)
                            ax.semilogy(times, tvds, alpha=0.5)
                            
                    # Theoretical bound
                    if 'theoretical' in data['mhk']:
                        delta = data['mhk']['theoretical']['delta']
                        t_max = max(times)
                        t_theory = np.linspace(1, t_max, 100)
                        tvd_theory = [(1 - delta)**t for t in t_theory]
                        ax.semilogy(t_theory, tvd_theory, 'k--', 
                                   label='Theoretical', linewidth=2)
                        
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('TVD to stationary')
                    ax.set_title(f'{lattice_key}, $\sigma$={sigma:.2f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
                    break
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'tvd_decay_comparison.pdf'))
        plt.close()
        
    def _plot_dimension_scaling(self, results):
        """Create dimension scaling plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filter out None values
        valid_idx = [i for i, t in enumerate(results['klein_times']) if t is not None]
        dims = [results['dimensions'][i] for i in valid_idx]
        
        # Plot 1: Computational time scaling
        ax = axes[0, 0]
        klein_times = [results['klein_times'][i] for i in valid_idx]
        mhk_times = [results['mhk_times'][i] for i in valid_idx]
        
        ax.loglog(dims, klein_times, 'b-o', label='Klein')
        ax.loglog(dims, mhk_times, 'r-s', label='MHK')
        ax.set_xlabel('Dimension $n$')
        ax.set_ylabel('Time per sample (s)')
        ax.set_title('Computational Time Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mixing time scaling
        ax = axes[0, 1]
        theoretical = [results['theoretical_mixing'][i] for i in valid_idx 
                      if results['theoretical_mixing'][i] is not None]
        empirical = [results['empirical_mixing'][i] for i in valid_idx
                    if results['empirical_mixing'][i] is not None]
        
        if theoretical:
            ax.loglog(dims[:len(theoretical)], theoretical, 'b-', 
                     label='Theoretical', linewidth=2)
        if empirical:
            ax.loglog(dims[:len(empirical)], empirical, 'ro', 
                     label='Empirical', markersize=8)
            
        # Fit polynomial
        if len(empirical) > 3:
            log_dims = np.log(dims[:len(empirical)])
            log_mixing = np.log(empirical)
            coeff = np.polyfit(log_dims, log_mixing, 1)[0]
            ax.text(0.05, 0.95, f'Scaling: $n^{{{coeff:.2f}}}$',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat'))
                   
        ax.set_xlabel('Dimension $n$')
        ax.set_ylabel('Mixing time')
        ax.set_title('Mixing Time Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Spectral gap
        ax = axes[1, 0]
        gaps = [results['spectral_gaps'][i] for i in valid_idx
               if results['spectral_gaps'][i] is not None]
        
        if gaps:
            ax.loglog(dims[:len(gaps)], gaps, 'g-^', linewidth=2)
            ax.set_xlabel('Dimension $n$')
            ax.set_ylabel('Spectral gap')
            ax.set_title('Spectral Gap vs Dimension')
            ax.grid(True, alpha=0.3)
            
        # Plot 4: Memory usage
        ax = axes[1, 1]
        memory = [results['memory_usage'][i] for i in valid_idx]
        
        ax.plot(dims, memory, 'k-d', linewidth=2)
        ax.set_xlabel('Dimension $n$')
        ax.set_ylabel('Memory usage (MB)')
        ax.set_title('Memory Scaling')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'dimension_scaling.pdf'))
        plt.close()
        
    def _generate_latex_tables(self, results):
        """Generate LaTeX tables for paper."""
        # Table 1: Algorithm comparison
        if 'algorithm_comparison' in results:
            self._generate_algorithm_table(results['algorithm_comparison'])
            
        # Table 2: Dimension scaling
        if 'dimension_scaling' in results:
            self._generate_scaling_table(results['dimension_scaling'])
            
        # Table 3: Parameter recommendations
        if 'sigma_sensitivity' in results:
            self._generate_parameter_table(results['sigma_sensitivity'])
            
    def _generate_algorithm_table(self, results):
        """Generate algorithm comparison table."""
        table = r"""\begin{table}[h]
\centering
\caption{Algorithm Comparison: Klein vs MH-Klein}
\begin{tabular}{lcccc}
\hline
Lattice & $\sigma$ & Klein (samples/s) & MHK Accept. Rate & MHK Mixing Time \\
\hline
"""
        
        for lattice_key, sigma_results in results.items():
            for sigma, data in list(sigma_results.items())[:3]:  # First 3 sigmas
                klein_speed = data['klein']['samples_per_second']
                mhk_accept = np.mean(data['mhk']['acceptance_rates'])
                mhk_mixing = data['mhk'].get('empirical_mixing_time', '--')
                
                table += f"{lattice_key} & {sigma:.2f} & {klein_speed:.1f} & "
                table += f"{mhk_accept:.3f} & {mhk_mixing} \\\\\n"
                
        table += r"""\hline
\end{tabular}
\end{table}"""
        
        with open(os.path.join(self.dirs['tables'], 'algorithm_comparison.tex'), 'w') as f:
            f.write(table)
            
    def _generate_readme(self):
        """Generate comprehensive README."""
        readme = f"""# Cryptographic Lattice Gaussian Sampling Experiments

## Experiment ID: {self.experiment_id}

## Overview
Comprehensive experimental study of lattice Gaussian sampling for cryptographic applications,
focusing on convergence diagnostics, sample quality, mixing time, and parameter sensitivity
for Falcon/NTRU applications.

## Setup Instructions

### Prerequisites
- Python 3.8+
- SageMath 9.0+
- Required packages: numpy, matplotlib, pandas, scipy

### Installation
```bash
pip install -r requirements.txt
sage -pip install [any additional sage packages]
```

### Running Experiments
```python
from cryptographic_experiments import CryptographicLatticeExperiments

# Initialize
exp = CryptographicLatticeExperiments(
    output_dir='results',
    n_cores=8,
    random_seed=42
)

# Generate lattices
lattices = exp.generate_lattice_suite()

# Run experiments
results = {{
    'algorithm_comparison': exp.compare_algorithms(lattices),
    'dimension_scaling': exp.dimension_scaling_experiment(),
    'sigma_sensitivity': exp.sigma_sensitivity(lattices[('ntru', 512, 'Falcon-512')]),
    # ... additional experiments
}}

# Generate outputs
exp.generate_all_outputs(results)
```

## Output Structure
- `figures/`: Publication-ready plots (PDF/SVG)
- `tables/`: LaTeX tables
- `data/`: Raw experimental data
- `logs/`: Experiment logs and metadata
- `checkpoints/`: Intermediate results

## Reproducibility
Random seed: {self.random_seed}
All parameters and seeds are logged for full reproducibility.

## Citation
If you use this code, please cite:
[Your paper reference here]
"""
        
        with open(os.path.join(self.output_dir, 'README.md'), 'w') as f:
            f.write(readme)
            

# Additional lattice implementations

class CheckerboardLattice(Lattice):
    """D_n checkerboard lattice implementation."""
    
    def __init__(self, n):
        super().__init__(n)
        self.name = f"D_{n}"
        self.generate_basis()
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Generate D_n basis vectors."""
        n = self.dimension
        basis_vectors = []
        
        # Vectors e_i - e_{i+1} for i = 0, ..., n-2
        for i in range(n - 1):
            v = vector(RDF, [0] * n)
            v[i] = 1
            v[i + 1] = -1
            basis_vectors.append(v)
            
        # Vector e_{n-2} + e_{n-1}
        v = vector(RDF, [0] * n)
        v[n - 2] = 1
        v[n - 1] = 1
        basis_vectors.append(v)
        
        self._basis = Matrix(RDF, basis_vectors)
        

class QaryLattice(Lattice):
    """q-ary lattice implementation."""
    
    def __init__(self, n, q, A=None):
        super().__init__(n)
        self.q = q
        self.A = A if A is not None else identity_matrix(ZZ, n)
        self.name = f"q-ary(n={n},q={q})"
        self.generate_basis()
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Generate q-ary lattice basis."""
        # Basis is [A | qI]^T in appropriate form
        self._basis = self.A.change_ring(RDF)


# Main execution
if __name__ == '__main__':
    # Quick test
    exp = CryptographicLatticeExperiments(n_cores=2)
    
    # Generate test lattices
    lattices = {
        ('checkerboard', 8): CheckerboardLattice(8),
        ('ntru', 256, 'test'): NTRULattice(256, 7681, 1.17)
    }
    
    # Run minimal test
    results = exp.compare_algorithms(
        lattices,
        sigma_values=[5.0, 10.0],
        n_samples=1000,
        n_chains=2
    )
    
    print("Test completed successfully!")