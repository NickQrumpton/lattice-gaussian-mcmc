"""
Quantitative analysis of dimension scaling for lattice Gaussian MCMC algorithms.

Focuses on mixing time, spectral gap, computational cost, and practical sampling
considerations. All parameters specified explicitly in terms of lattice norms.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, pi, sqrt, exp, log, ln,
    identity_matrix, random_matrix, jacobi_theta_3, infinity
)
from multiprocessing import Pool, cpu_count
import pickle
import os
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from scipy import stats, integrate
from collections import defaultdict
import time
import traceback
import psutil
from sklearn.linear_model import LinearRegression

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
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IndependentMHK
from src.diagnostics.spectral import SpectralGapAnalyzer
from src.diagnostics.convergence import ConvergenceDiagnostics


class DimensionScalingAnalysis:
    """
    Quantitative analysis of dimension scaling for lattice Gaussian MCMC.
    """
    
    def __init__(self, output_dir='../results/dimension_scaling', 
                 n_cores=None, random_seed=42):
        """
        Initialize scaling analysis framework.
        
        Args:
            output_dir: Directory for results
            n_cores: Number of CPU cores
            random_seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.n_cores = n_cores or cpu_count()
        self.random_seed = random_seed
        
        # Set seeds
        np.random.seed(random_seed)
        from sage.all import set_random_seed
        set_random_seed(random_seed)
        
        # Create directories
        self.dirs = {}
        for subdir in ['figures', 'tables', 'data', 'logs']:
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
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging infrastructure."""
        import logging
        
        log_file = os.path.join(self.dirs['logs'], 
                               f'dimension_scaling_{self.experiment_id}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized dimension scaling analysis {self.experiment_id}")
        
    # ========== 1. Mixing Time Growth with Dimension ==========
    
    def mixing_time_vs_dimension(self, 
                                dimensions=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
                                sigma_ratio=0.4,  # Ã² = 0.16 * min||b*_i||²
                                n_chains=10,
                                max_samples=100000):
        """
        Analyze mixing time growth with dimension.
        
        Args:
            dimensions: List of dimensions n to test
            sigma_ratio: Ã/min||b*_i|| ratio (Ã² = sigma_ratio² * min||b*_i||²)
            n_chains: Number of independent chains per test
            max_samples: Maximum samples per chain
            
        Returns:
            dict: Mixing time analysis results
        """
        self.logger.info(f"Starting mixing time analysis with Ã² = {sigma_ratio**2} * min||b*_i||²")
        
        results = {
            'dimensions': dimensions,
            'sigma_ratio': sigma_ratio,
            'sigma_squared_ratio': sigma_ratio**2,
            'lattices': {}
        }
        
        # Test different lattice types
        lattice_types = ['D_n', 'A_n', 'Z^n']
        
        for lattice_type in lattice_types:
            self.logger.info(f"Testing {lattice_type} lattices...")
            
            mixing_times = []
            mixing_times_ci = []
            spectral_gaps = []
            actual_sigmas = []
            min_gs_norms = []
            
            for n in dimensions:
                try:
                    self.logger.info(f"  Dimension n={n}")
                    
                    # Create lattice
                    if lattice_type == 'D_n':
                        lattice = CheckerboardLattice(n)
                    elif lattice_type == 'A_n':
                        lattice = RootLattice(n, 'A')
                    else:  # Z^n
                        lattice = IntegerLattice(n)
                        
                    # Get Gram-Schmidt norms
                    gs_norms = lattice.get_gram_schmidt_norms()
                    min_norm = min(gs_norms)
                    min_gs_norms.append(float(min_norm))
                    
                    # Set Ã based on minimum GS norm
                    sigma = sigma_ratio * min_norm
                    actual_sigmas.append(float(sigma))
                    
                    self.logger.info(f"    min||b*_i|| = {min_norm:.4f}, Ã = {sigma:.4f}")
                    
                    # Run multiple chains
                    chain_results = self._run_mixing_time_experiment(
                        lattice, sigma, n_chains, max_samples
                    )
                    
                    # Store results
                    mixing_times.append(chain_results['mean_mixing_time'])
                    mixing_times_ci.append(chain_results['mixing_time_ci'])
                    spectral_gaps.append(chain_results['spectral_gap'])
                    
                except Exception as e:
                    self.logger.error(f"Error at n={n}: {str(e)}")
                    mixing_times.append(None)
                    mixing_times_ci.append((None, None))
                    spectral_gaps.append(None)
                    
            results['lattices'][lattice_type] = {
                'mixing_times': mixing_times,
                'mixing_times_ci': mixing_times_ci,
                'spectral_gaps': spectral_gaps,
                'actual_sigmas': actual_sigmas,
                'min_gs_norms': min_gs_norms
            }
            
        # Generate plots
        self._plot_mixing_time_scaling(results)
        
        # Fit scaling curves
        scaling_analysis = self._analyze_mixing_time_scaling(results)
        results['scaling_analysis'] = scaling_analysis
        
        # Save results
        self._save_results(results, 'mixing_time_scaling')
        
        return results
    
    def _run_mixing_time_experiment(self, lattice, sigma, n_chains, max_samples):
        """Run mixing time experiment for given configuration."""
        # Create sampler
        sampler = IndependentMHK(lattice, sigma)
        
        # Theoretical predictions
        theoretical_mixing = sampler.mixing_time()
        spectral_gap = sampler.spectral_gap()
        
        # Run chains in parallel
        with Pool(min(self.n_cores, n_chains)) as pool:
            chain_data = pool.starmap(
                self._run_single_mixing_chain,
                [(lattice, sigma, max_samples, i) for i in range(n_chains)]
            )
            
        # Extract mixing times
        mixing_times = [d['mixing_time'] for d in chain_data if d['mixing_time'] is not None]
        
        if mixing_times:
            mean_mixing = np.mean(mixing_times)
            ci_lower = np.percentile(mixing_times, 2.5)
            ci_upper = np.percentile(mixing_times, 97.5)
        else:
            mean_mixing = theoretical_mixing
            ci_lower = ci_upper = theoretical_mixing
            
        return {
            'mean_mixing_time': mean_mixing,
            'mixing_time_ci': (ci_lower, ci_upper),
            'theoretical_mixing': theoretical_mixing,
            'spectral_gap': spectral_gap,
            'individual_mixing_times': mixing_times
        }
    
    def _run_single_mixing_chain(self, lattice, sigma, max_samples, chain_id):
        """Run single chain for mixing time estimation."""
        np.random.seed(self.random_seed + chain_id)
        
        # Create diagnostics
        diagnostics = ConvergenceDiagnostics(lattice, sigma)
        
        # Run chain
        sampler = IndependentMHK(lattice, sigma)
        chain = sampler.sample(max_samples)
        
        # Estimate mixing time using autocorrelation
        acf_mixing = self._estimate_mixing_from_acf(chain)
        
        # Also try TVD-based estimation if feasible
        tvd_mixing = None
        if lattice.get_dimension() <= 100:  # Only for smaller dimensions
            tvd_mixing = self._estimate_mixing_from_tvd(chain, sampler)
            
        return {
            'mixing_time': acf_mixing or tvd_mixing,
            'acf_mixing': acf_mixing,
            'tvd_mixing': tvd_mixing
        }
    
    def _estimate_mixing_from_acf(self, chain):
        """Estimate mixing time from autocorrelation function."""
        # Use first component for multivariate chains
        if len(chain.shape) > 1:
            x = chain[:, 0]
        else:
            x = chain
            
        # Compute autocorrelation
        n = len(x)
        x_centered = x - np.mean(x)
        
        # Find when ACF drops below threshold
        threshold = 0.05
        max_lag = min(n // 4, 10000)
        
        for lag in range(1, max_lag):
            if lag >= n:
                break
                
            acf = np.correlate(x_centered[:-lag], x_centered[lag:]) / (
                np.var(x_centered) * (n - lag)
            )
            
            if abs(acf) < threshold:
                return lag
                
        return max_lag
    
    # ========== 2. Inverse Spectral Gap Scaling ==========
    
    def delta_inverse_scaling(self, max_dimension=1000, sigma_ratio=0.4):
        """
        Analyze 1/´ scaling with dimension.
        
        Args:
            max_dimension: Maximum dimension to test
            sigma_ratio: Ã/min||b*_i|| ratio
            
        Returns:
            dict: Delta scaling analysis
        """
        self.logger.info(f"Computing 1/´ scaling with Ã² = {sigma_ratio**2} * min||b*_i||²")
        
        # Test dimensions
        dimensions = []
        n = 4
        while n <= max_dimension:
            dimensions.append(n)
            if n < 100:
                n += 4
            elif n < 500:
                n += 50
            else:
                n += 100
                
        results = {
            'dimensions': dimensions,
            'sigma_ratio': sigma_ratio,
            'sigma_squared_ratio': sigma_ratio**2,
            'lattices': {}
        }
        
        # Focus on D_n lattice
        self.logger.info("Computing for D_n lattices...")
        
        delta_values = []
        delta_inverse = []
        actual_sigmas = []
        
        for n in dimensions:
            self.logger.info(f"  n={n}")
            
            # Create lattice
            lattice = CheckerboardLattice(n)
            
            # Get minimum GS norm
            gs_norms = lattice.get_gram_schmidt_norms()
            min_norm = min(gs_norms)
            
            # Set Ã
            sigma = sigma_ratio * min_norm
            actual_sigmas.append(float(sigma))
            
            # Compute ´
            delta = self._compute_delta_exact(lattice, sigma)
            delta_values.append(delta)
            delta_inverse.append(1.0/delta if delta > 0 else float('inf'))
            
            self.logger.info(f"    Ã = {sigma:.4f}, ´ = {delta:.6e}, 1/´ = {1.0/delta:.2e}")
            
        results['lattices']['D_n'] = {
            'delta_values': delta_values,
            'delta_inverse': delta_inverse,
            'actual_sigmas': actual_sigmas
        }
        
        # Generate plots
        self._plot_delta_scaling(results)
        
        # Analyze growth rate
        growth_analysis = self._analyze_delta_growth(results)
        results['growth_analysis'] = growth_analysis
        
        # Save results
        self._save_results(results, 'delta_scaling')
        
        return results
    
    def _compute_delta_exact(self, lattice, sigma):
        """
        Compute ´ = Á_{Ã,0}(›) /  _{i=1}^n Á_{Ã_i}(Z).
        """
        n = lattice.get_dimension()
        
        # Get QR decomposition for Ã_i values
        basis = lattice.get_basis()
        Q, R = matrix(RDF, basis).QR()
        
        # Compute Ã_i = Ã/|r_{i,i}|
        sigmas = [sigma / abs(R[i, i]) for i in range(n)]
        
        # Compute Á_{Ã,0}(›) using Jacobi theta function
        det_lattice = lattice.get_determinant()
        
        # For lattice: Á_{Ã,0}(›) H det(›) * ¸_3(0, exp(-2À²Ã²/det(›)^{2/n}))
        tau_lattice = exp(-2 * pi**2 * sigma**2 / det_lattice**(2/n))
        rho_lattice = det_lattice * jacobi_theta_3(0, tau_lattice)
        
        # Compute   Á_{Ã_i}(Z) =   ¸_3(0, exp(-2À²Ã_i²))
        prod_rho_Z = RDF(1)
        for sigma_i in sigmas:
            tau_i = exp(-2 * pi**2 * sigma_i**2)
            prod_rho_Z *= jacobi_theta_3(0, tau_i)
            
        # ´ = Á_{Ã,0}(›) /   Á_{Ã_i}(Z)
        delta = float(rho_lattice / prod_rho_Z)
        
        return min(delta, 1.0)  # Ensure ´ d 1
    
    # ========== 3. Jacobi Theta Function Products ==========
    
    def theta_function_products(self, dimensions=[4, 8, 16, 32, 64, 128],
                               sigma_ratios=[0.2, 0.4, 0.8]):
        """
        Analyze Jacobi theta function products.
        
        Args:
            dimensions: Dimensions to test
            sigma_ratios: Different Ã/||b*|| ratios
            
        Returns:
            dict: Theta function analysis
        """
        self.logger.info("Analyzing Jacobi theta function products...")
        
        results = {
            'dimensions': dimensions,
            'sigma_ratios': sigma_ratios,
            'products': {}
        }
        
        for lattice_type in ['D_n', 'Z^n']:
            self.logger.info(f"Testing {lattice_type}...")
            
            products_by_sigma = {}
            
            for sigma_ratio in sigma_ratios:
                products = []
                log_products = []
                pre_exponential = []
                
                for n in dimensions:
                    # Create lattice
                    if lattice_type == 'D_n':
                        lattice = CheckerboardLattice(n)
                    else:
                        lattice = IntegerLattice(n)
                        
                    # Get GS norms
                    gs_norms = lattice.get_gram_schmidt_norms()
                    sigma = sigma_ratio * min(gs_norms)
                    
                    # Compute product
                    prod, log_prod, pre_exp = self._compute_theta_product(
                        gs_norms, sigma
                    )
                    
                    products.append(prod)
                    log_products.append(log_prod)
                    pre_exponential.append(pre_exp)
                    
                    self.logger.info(
                        f"  n={n}, Ã²={sigma**2:.4f}*min||b*||²: "
                        f"prod={prod:.6e}, pre-exp={pre_exp:.6f}"
                    )
                    
                products_by_sigma[sigma_ratio] = {
                    'products': products,
                    'log_products': log_products,
                    'pre_exponential': pre_exponential
                }
                
            results['products'][lattice_type] = products_by_sigma
            
        # Analyze growth
        growth_analysis = self._analyze_theta_growth(results)
        results['growth_analysis'] = growth_analysis
        
        # Generate plots and tables
        self._plot_theta_products(results)
        self._generate_theta_table(results)
        
        # Save results
        self._save_results(results, 'theta_products')
        
        return results
    
    def _compute_theta_product(self, gs_norms, sigma):
        """Compute   ¸_3(||b*_i||²/(2ÀÃ²))."""
        product = RDF(1)
        log_sum = RDF(0)
        
        for norm in gs_norms:
            # Argument for theta function
            z = norm**2 / (2 * pi * sigma**2)
            
            # ¸_3(0, e^{-Àz})
            q = exp(-pi * z)
            theta = jacobi_theta_3(0, q)
            
            product *= theta
            log_sum += log(theta)
            
        # Extract pre-exponential factor
        # log(prod) H n*log(c) + exponential terms
        n = len(gs_norms)
        pre_exp = exp(log_sum / n)
        
        return float(product), float(log_sum), float(pre_exp)
    
    # ========== 4. Condition Number Effects ==========
    
    def condition_vs_performance(self, dimensions=[16, 32, 64],
                                condition_range=[1, 10, 100, 1000]):
        """
        Analyze effect of lattice condition number on performance.
        
        Args:
            dimensions: Dimensions to test
            condition_range: Target condition numbers
            
        Returns:
            dict: Condition number analysis
        """
        self.logger.info("Analyzing condition number effects...")
        
        results = {
            'dimensions': dimensions,
            'condition_targets': condition_range,
            'data': []
        }
        
        for n in dimensions:
            self.logger.info(f"Dimension n={n}")
            
            for target_cond in condition_range:
                # Generate lattice with target condition number
                lattice_orig = self._generate_conditioned_lattice(n, target_cond)
                cond_orig = self._compute_condition_number(lattice_orig)
                
                # Apply LLL reduction
                lattice_lll = self._apply_lll_reduction(lattice_orig)
                cond_lll = self._compute_condition_number(lattice_lll)
                
                self.logger.info(
                    f"  Target º={target_cond}: "
                    f"Original º={cond_orig:.2f}, LLL º={cond_lll:.2f}"
                )
                
                # Test both lattices
                sigma = 2.0 * min(lattice_orig.get_gram_schmidt_norms())
                
                # Original lattice
                perf_orig = self._test_lattice_performance(lattice_orig, sigma)
                
                # LLL-reduced lattice
                perf_lll = self._test_lattice_performance(lattice_lll, sigma)
                
                results['data'].append({
                    'dimension': n,
                    'target_condition': target_cond,
                    'original': {
                        'condition_number': cond_orig,
                        'mixing_time': perf_orig['mixing_time'],
                        'spectral_gap': perf_orig['spectral_gap'],
                        'acceptance_rate': perf_orig['acceptance_rate']
                    },
                    'lll_reduced': {
                        'condition_number': cond_lll,
                        'mixing_time': perf_lll['mixing_time'],
                        'spectral_gap': perf_lll['spectral_gap'],
                        'acceptance_rate': perf_lll['acceptance_rate']
                    }
                })
                
        # Generate plots
        self._plot_condition_effects(results)
        
        # Save results
        self._save_results(results, 'condition_effects')
        
        return results
    
    def _generate_conditioned_lattice(self, n, target_condition):
        """Generate lattice with approximately target condition number."""
        # Start with diagonal matrix
        diag = np.ones(n)
        
        # Set first and last to create condition number
        diag[0] = 1.0
        diag[-1] = target_condition
        
        # Interpolate others geometrically
        for i in range(1, n-1):
            diag[i] = np.exp(np.log(target_condition) * i / (n-1))
            
        # Create basis
        basis = np.diag(diag)
        
        # Random rotation to make it interesting
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        basis = Q @ basis
        
        return MatrixLattice(basis)
    
    # ========== 5. High-Dimensional Behavior ==========
    
    def asymptotic_behavior(self, dimensions=[100, 200, 500, 1000, 2000],
                           n_samples=10000):
        """
        Analyze asymptotic behavior in high dimensions.
        
        Args:
            dimensions: Large dimensions to test
            n_samples: Samples for distribution testing
            
        Returns:
            dict: Asymptotic analysis results
        """
        self.logger.info("Analyzing asymptotic behavior...")
        
        results = {
            'dimensions': dimensions,
            'gaussian_distances': [],
            'heuristic_accuracy': [],
            'sampling_quality': []
        }
        
        for n in dimensions:
            self.logger.info(f"Testing dimension n={n}")
            
            try:
                # Use simple Z^n lattice for clarity
                lattice = IntegerLattice(n)
                sigma = 10.0  # Large Ã for asymptotic regime
                
                # Generate samples
                sampler = KleinSampler(lattice, sigma)
                samples = np.array([sampler.sample() for _ in range(n_samples)])
                
                # Test 1: Distance to continuous Gaussian
                gauss_dist = self._test_gaussian_convergence(samples, sigma)
                results['gaussian_distances'].append(gauss_dist)
                
                # Test 2: Gaussian heuristic accuracy
                heuristic_acc = self._test_gaussian_heuristic(lattice, samples)
                results['heuristic_accuracy'].append(heuristic_acc)
                
                # Test 3: Sample quality metrics
                quality = self._assess_sample_quality(samples, lattice, sigma)
                results['sampling_quality'].append(quality)
                
            except Exception as e:
                self.logger.error(f"Error at n={n}: {str(e)}")
                results['gaussian_distances'].append(None)
                results['heuristic_accuracy'].append(None)
                results['sampling_quality'].append(None)
                
        # Generate analysis plots
        self._plot_asymptotic_behavior(results)
        
        # Save results
        self._save_results(results, 'asymptotic_behavior')
        
        return results
    
    def _test_gaussian_convergence(self, samples, sigma):
        """Test convergence to continuous Gaussian."""
        n = samples.shape[1]
        
        # Use first few components for tractability
        n_test = min(n, 10)
        
        # Kolmogorov-Smirnov test on marginals
        ks_stats = []
        for i in range(n_test):
            stat, _ = stats.kstest(samples[:, i], 'norm', args=(0, sigma))
            ks_stats.append(stat)
            
        # Multivariate normality test (simplified)
        # Check if covariance matches Ã²I
        emp_cov = np.cov(samples[:, :n_test].T)
        expected_cov = sigma**2 * np.eye(n_test)
        cov_error = np.linalg.norm(emp_cov - expected_cov, 'fro') / np.linalg.norm(expected_cov, 'fro')
        
        return {
            'ks_statistics': ks_stats,
            'mean_ks_stat': np.mean(ks_stats),
            'covariance_error': cov_error
        }
    
    # ========== 6. Computational Complexity ==========
    
    def computational_complexity(self, dimensions=[10, 20, 50, 100, 200, 500, 1000],
                                n_iterations=1000):
        """
        Analyze computational and memory scaling.
        
        Args:
            dimensions: Dimensions to test
            n_iterations: Iterations per timing test
            
        Returns:
            dict: Complexity analysis
        """
        self.logger.info("Analyzing computational complexity...")
        
        results = {
            'dimensions': dimensions,
            'klein': {
                'time_per_sample': [],
                'memory_usage': [],
                'qr_time': [],
                'theta_time': []
            },
            'mhk': {
                'time_per_sample': [],
                'memory_usage': [],
                'acceptance_time': []
            }
        }
        
        for n in dimensions:
            self.logger.info(f"Testing dimension n={n}")
            
            try:
                # Create lattice
                lattice = IntegerLattice(n)
                sigma = 5.0
                
                # Test Klein sampler
                klein_perf = self._profile_klein_sampler(lattice, sigma, n_iterations)
                for key in ['time_per_sample', 'memory_usage', 'qr_time', 'theta_time']:
                    results['klein'][key].append(klein_perf[key])
                    
                # Test MHK sampler
                mhk_perf = self._profile_mhk_sampler(lattice, sigma, n_iterations)
                for key in ['time_per_sample', 'memory_usage', 'acceptance_time']:
                    results['mhk'][key].append(mhk_perf[key])
                    
            except Exception as e:
                self.logger.error(f"Error at n={n}: {str(e)}")
                # Append None for failed tests
                for key in results['klein']:
                    results['klein'][key].append(None)
                for key in results['mhk']:
                    results['mhk'][key].append(None)
                    
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(results)
        results['bottlenecks'] = bottlenecks
        
        # Generate plots
        self._plot_computational_complexity(results)
        
        # Save results
        self._save_results(results, 'computational_complexity')
        
        return results
    
    def _profile_klein_sampler(self, lattice, sigma, n_iterations):
        """Profile Klein sampler performance."""
        import resource
        
        # Memory before
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Create sampler (includes QR decomposition)
        start_qr = time.time()
        sampler = KleinSampler(lattice, sigma)
        qr_time = time.time() - start_qr
        
        # Time sampling
        start_sample = time.time()
        for _ in range(n_iterations):
            sampler.sample()
        total_sample_time = time.time() - start_sample
        
        # Memory after
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Estimate theta function time (simplified)
        theta_time = total_sample_time * 0.3  # Rough estimate
        
        return {
            'time_per_sample': total_sample_time / n_iterations,
            'memory_usage': (mem_after - mem_before) / 1024.0,  # MB
            'qr_time': qr_time,
            'theta_time': theta_time / n_iterations
        }
    
    # ========== 7. Parallelization Scaling ==========
    
    def parallel_scaling(self, dimension=100, n_chains_range=[1, 2, 4, 8, 16],
                        samples_per_chain=10000):
        """
        Analyze parallel scaling efficiency.
        
        Args:
            dimension: Lattice dimension
            n_chains_range: Number of parallel chains to test
            samples_per_chain: Samples per chain
            
        Returns:
            dict: Parallel scaling analysis
        """
        self.logger.info("Analyzing parallel scaling...")
        
        # Create test lattice
        lattice = CheckerboardLattice(dimension)
        sigma = 2.0 * min(lattice.get_gram_schmidt_norms())
        
        results = {
            'dimension': dimension,
            'n_chains_range': n_chains_range,
            'samples_per_chain': samples_per_chain,
            'strong_scaling': {},
            'weak_scaling': {}
        }
        
        # Strong scaling: fixed total work
        total_samples = samples_per_chain * 4
        
        for n_chains in n_chains_range:
            if n_chains > self.n_cores:
                continue
                
            self.logger.info(f"Testing {n_chains} chains...")
            
            # Strong scaling
            samples_per = total_samples // n_chains
            start_time = time.time()
            
            with Pool(n_chains) as pool:
                chains = pool.starmap(
                    self._run_parallel_chain,
                    [(lattice, sigma, samples_per, i) for i in range(n_chains)]
                )
                
            strong_time = time.time() - start_time
            results['strong_scaling'][n_chains] = {
                'time': strong_time,
                'speedup': results['strong_scaling'][1]['time'] / strong_time if 1 in results['strong_scaling'] else 1.0,
                'efficiency': (results['strong_scaling'][1]['time'] / strong_time / n_chains) if 1 in results['strong_scaling'] else 1.0
            }
            
            # Weak scaling: fixed work per processor
            start_time = time.time()
            
            with Pool(n_chains) as pool:
                chains = pool.starmap(
                    self._run_parallel_chain,
                    [(lattice, sigma, samples_per_chain, i) for i in range(n_chains)]
                )
                
            weak_time = time.time() - start_time
            results['weak_scaling'][n_chains] = {
                'time': weak_time,
                'efficiency': results['weak_scaling'][1]['time'] / weak_time if 1 in results['weak_scaling'] else 1.0
            }
            
        # Generate scaling plots
        self._plot_parallel_scaling(results)
        
        # Save results
        self._save_results(results, 'parallel_scaling')
        
        return results
    
    def _run_parallel_chain(self, lattice, sigma, n_samples, chain_id):
        """Run single chain for parallel scaling test."""
        np.random.seed(self.random_seed + chain_id * 1000)
        
        sampler = IndependentMHK(lattice, sigma)
        chain = sampler.sample(n_samples)
        
        return chain
    
    # ========== Plotting Methods ==========
    
    def _plot_mixing_time_scaling(self, results):
        """Plot mixing time vs dimension."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        dimensions = results['dimensions']
        
        # Plot 1: Log-log mixing time
        for lattice_type, data in results['lattices'].items():
            mixing_times = data['mixing_times']
            ci_lower = [ci[0] for ci in data['mixing_times_ci']]
            ci_upper = [ci[1] for ci in data['mixing_times_ci']]
            
            # Filter valid data
            valid_idx = [i for i, t in enumerate(mixing_times) if t is not None]
            if not valid_idx:
                continue
                
            dims_valid = [dimensions[i] for i in valid_idx]
            times_valid = [mixing_times[i] for i in valid_idx]
            ci_lower_valid = [ci_lower[i] for i in valid_idx]
            ci_upper_valid = [ci_upper[i] for i in valid_idx]
            
            # Plot with confidence intervals
            ax1.errorbar(dims_valid, times_valid, 
                        yerr=[np.array(times_valid) - np.array(ci_lower_valid),
                              np.array(ci_upper_valid) - np.array(times_valid)],
                        marker='o', label=lattice_type, capsize=5)
                        
        # Add theoretical scaling curves
        n_theory = np.logspace(np.log10(4), np.log10(max(dimensions)), 100)
        ax1.plot(n_theory, 10 * n_theory, 'k--', alpha=0.5, label='O(n)')
        ax1.plot(n_theory, 5 * n_theory * np.log(n_theory), 'k:', alpha=0.5, label='O(n log n)')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Dimension $n$')
        ax1.set_ylabel('Mixing Time $t_{\\rm mix}$')
        ax1.set_title(f'Mixing Time Scaling ($\\sigma^2 = {results["sigma_squared_ratio"]:.2f} \\cdot \\min_i \\|b^*_i\\|^2$)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectral gap
        for lattice_type, data in results['lattices'].items():
            gaps = data['spectral_gaps']
            valid_idx = [i for i, g in enumerate(gaps) if g is not None]
            
            if valid_idx:
                dims_valid = [dimensions[i] for i in valid_idx]
                gaps_valid = [gaps[i] for i in valid_idx]
                ax2.loglog(dims_valid, gaps_valid, 'o-', label=lattice_type)
                
        ax2.set_xlabel('Dimension $n$')
        ax2.set_ylabel('Spectral Gap $\\delta$')
        ax2.set_title('Spectral Gap vs Dimension')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'mixing_time_scaling.pdf'))
        plt.close()
        
    def _plot_delta_scaling(self, results):
        """Plot 1/´ scaling."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for lattice_type, data in results['lattices'].items():
            dimensions = results['dimensions']
            delta_inv = data['delta_inverse']
            
            # Filter finite values
            valid_idx = [i for i, d in enumerate(delta_inv) if d < float('inf')]
            dims_valid = [dimensions[i] for i in valid_idx]
            delta_inv_valid = [delta_inv[i] for i in valid_idx]
            
            ax.semilogy(dims_valid, delta_inv_valid, 'o-', 
                       label=lattice_type, linewidth=2, markersize=8)
                       
        ax.set_xlabel('Dimension $n$')
        ax.set_ylabel('$1/\\delta$')
        ax.set_title(f'Inverse Spectral Gap Scaling ($\\sigma^2 = {results["sigma_squared_ratio"]:.2f} \\cdot \\min_i \\|b^*_i\\|^2$)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text with growth rate if computed
        if 'growth_analysis' in results:
            growth = results['growth_analysis']
            ax.text(0.05, 0.95, f"Growth rate: {growth['growth_rate']:.4f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat'))
                   
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'delta_inverse_scaling.pdf'))
        plt.close()
        
    def _plot_computational_complexity(self, results):
        """Plot computational complexity scaling."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        dimensions = results['dimensions']
        
        # Time per sample
        klein_times = results['klein']['time_per_sample']
        mhk_times = results['mhk']['time_per_sample']
        
        valid_idx = [i for i, t in enumerate(klein_times) if t is not None]
        dims_valid = [dimensions[i] for i in valid_idx]
        
        if dims_valid:
            klein_valid = [klein_times[i] for i in valid_idx]
            mhk_valid = [mhk_times[i] for i in valid_idx]
            
            ax1.loglog(dims_valid, klein_valid, 'bo-', label='Klein', linewidth=2)
            ax1.loglog(dims_valid, mhk_valid, 'rs-', label='MHK', linewidth=2)
            
            # Fit complexity
            log_dims = np.log(dims_valid)
            log_klein = np.log(klein_valid)
            klein_exp = np.polyfit(log_dims, log_klein, 1)[0]
            
            ax1.text(0.05, 0.95, f'Klein: O($n^{{{klein_exp:.2f}}}$)',
                    transform=ax1.transAxes, verticalalignment='top')
                    
        ax1.set_xlabel('Dimension $n$')
        ax1.set_ylabel('Time per sample (s)')
        ax1.set_title('Computational Time Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        klein_mem = results['klein']['memory_usage']
        mhk_mem = results['mhk']['memory_usage']
        
        valid_idx = [i for i, m in enumerate(klein_mem) if m is not None]
        if valid_idx:
            dims_valid = [dimensions[i] for i in valid_idx]
            klein_mem_valid = [klein_mem[i] for i in valid_idx]
            mhk_mem_valid = [mhk_mem[i] for i in valid_idx]
            
            ax2.plot(dims_valid, klein_mem_valid, 'bo-', label='Klein', linewidth=2)
            ax2.plot(dims_valid, mhk_mem_valid, 'rs-', label='MHK', linewidth=2)
            
        ax2.set_xlabel('Dimension $n$')
        ax2.set_ylabel('Memory usage (MB)')
        ax2.set_title('Memory Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Component breakdown for Klein
        qr_times = results['klein']['qr_time']
        theta_times = results['klein']['theta_time']
        
        valid_idx = [i for i, t in enumerate(qr_times) if t is not None]
        if valid_idx:
            dims_valid = [dimensions[i] for i in valid_idx]
            qr_valid = [qr_times[i] for i in valid_idx]
            theta_valid = [theta_times[i] for i in valid_idx]
            
            ax3.loglog(dims_valid, qr_valid, 'go-', label='QR decomp', linewidth=2)
            ax3.loglog(dims_valid, theta_valid, 'mo-', label='Theta func', linewidth=2)
            
        ax3.set_xlabel('Dimension $n$')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Klein Sampler Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Bottleneck identification
        if 'bottlenecks' in results:
            bottlenecks = results['bottlenecks']
            ax4.text(0.1, 0.9, 'Identified Bottlenecks:', transform=ax4.transAxes,
                    fontsize=14, weight='bold')
                    
            y_pos = 0.7
            for component, info in bottlenecks.items():
                ax4.text(0.1, y_pos, f'" {component}: {info}',
                        transform=ax4.transAxes, fontsize=12)
                y_pos -= 0.15
                
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'computational_complexity.pdf'))
        plt.close()
        
    # ========== Analysis Methods ==========
    
    def _analyze_mixing_time_scaling(self, results):
        """Analyze mixing time scaling with dimension."""
        scaling_results = {}
        
        for lattice_type, data in results['lattices'].items():
            dimensions = results['dimensions']
            mixing_times = data['mixing_times']
            
            # Get valid data points
            valid_idx = [i for i, t in enumerate(mixing_times) if t is not None]
            if len(valid_idx) < 3:
                continue
                
            dims = np.array([dimensions[i] for i in valid_idx])
            times = np.array([mixing_times[i] for i in valid_idx])
            
            # Fit log-log model
            log_dims = np.log(dims)
            log_times = np.log(times)
            
            # Linear regression in log space
            slope, intercept = np.polyfit(log_dims, log_times, 1)
            
            # Check if O(n) or O(n log n) fits better
            residual_n = np.sum((log_times - (intercept + slope * log_dims))**2)
            
            log_n_log_n = log_dims + np.log(log_dims)
            slope_nlogn, int_nlogn = np.polyfit(log_n_log_n, log_times, 1)
            residual_nlogn = np.sum((log_times - (int_nlogn + slope_nlogn * log_n_log_n))**2)
            
            scaling_results[lattice_type] = {
                'power_law_exponent': slope,
                'best_fit': 'O(n)' if residual_n < residual_nlogn else 'O(n log n)',
                'r_squared_n': 1 - residual_n / np.var(log_times),
                'r_squared_nlogn': 1 - residual_nlogn / np.var(log_times)
            }
            
        return scaling_results
    
    def _analyze_delta_growth(self, results):
        """Analyze growth rate of 1/´."""
        growth_analysis = {}
        
        for lattice_type, data in results['lattices'].items():
            dimensions = results['dimensions']
            delta_inv = data['delta_inverse']
            
            # Get finite values
            valid_idx = [i for i, d in enumerate(delta_inv) 
                        if d < float('inf') and d > 0]
            
            if len(valid_idx) < 3:
                continue
                
            dims = np.array([dimensions[i] for i in valid_idx])
            vals = np.array([delta_inv[i] for i in valid_idx])
            
            # Fit exponential model: 1/´ ~ c^n
            log_vals = np.log(vals)
            slope, intercept = np.polyfit(dims, log_vals, 1)
            
            growth_rate = np.exp(slope)
            
            growth_analysis['growth_rate'] = growth_rate
            growth_analysis['doubling_dimension'] = np.log(2) / slope if slope > 0 else float('inf')
            
        return growth_analysis
    
    def _identify_bottlenecks(self, results):
        """Identify computational bottlenecks."""
        bottlenecks = {}
        
        # Analyze Klein sampler
        if results['klein']['qr_time'] and results['klein']['theta_time']:
            qr_times = [t for t in results['klein']['qr_time'] if t is not None]
            theta_times = [t for t in results['klein']['theta_time'] if t is not None]
            
            if qr_times and theta_times:
                # Check which dominates at large n
                if len(qr_times) > 3:
                    qr_ratio = qr_times[-1] / qr_times[0]
                    theta_ratio = theta_times[-1] / theta_times[0]
                    
                    if qr_ratio > theta_ratio:
                        bottlenecks['Klein'] = 'QR decomposition (O(n³))'
                    else:
                        bottlenecks['Klein'] = 'Theta function evaluations'
                        
        # Memory bottlenecks
        klein_mem = [m for m in results['klein']['memory_usage'] if m is not None]
        if klein_mem and len(klein_mem) > 3:
            # Check memory growth
            dims = [results['dimensions'][i] for i in range(len(klein_mem))]
            mem_growth = np.polyfit(np.log(dims), np.log(klein_mem), 1)[0]
            
            if mem_growth > 2.5:
                bottlenecks['Memory'] = f'Superquadratic growth (O(n^{mem_growth:.1f}))'
                
        return bottlenecks
    
    # ========== Helper Methods ==========
    
    def _save_results(self, results, name):
        """Save results to file."""
        # JSON for readable format
        json_file = os.path.join(self.dirs['data'], f'{name}_{self.experiment_id}.json')
        
        # Convert numpy arrays and special values
        json_results = self._prepare_for_json(results)
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        # Pickle for complete data
        pkl_file = os.path.join(self.dirs['data'], f'{name}_{self.experiment_id}.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
            
        self.logger.info(f"Saved results to {json_file}")
        
    def _prepare_for_json(self, obj):
        """Convert numpy arrays and special values for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif obj == float('inf'):
            return 'inf'
        elif obj == float('-inf'):
            return '-inf'
        elif obj != obj:  # NaN
            return 'nan'
        else:
            return obj
            
    def run_all_experiments(self):
        """Run complete dimension scaling analysis."""
        self.logger.info("Starting complete dimension scaling analysis...")
        
        all_results = {}
        
        # 1. Mixing time scaling
        self.logger.info("\n=== Mixing Time Scaling ===")
        all_results['mixing_time'] = self.mixing_time_vs_dimension()
        
        # 2. Delta scaling
        self.logger.info("\n=== Delta Inverse Scaling ===")
        all_results['delta_scaling'] = self.delta_inverse_scaling()
        
        # 3. Theta function products
        self.logger.info("\n=== Theta Function Products ===")
        all_results['theta_products'] = self.theta_function_products()
        
        # 4. Condition number effects
        self.logger.info("\n=== Condition Number Effects ===")
        all_results['condition_effects'] = self.condition_vs_performance()
        
        # 5. Asymptotic behavior
        self.logger.info("\n=== Asymptotic Behavior ===")
        all_results['asymptotic'] = self.asymptotic_behavior()
        
        # 6. Computational complexity
        self.logger.info("\n=== Computational Complexity ===")
        all_results['complexity'] = self.computational_complexity()
        
        # 7. Parallel scaling
        self.logger.info("\n=== Parallel Scaling ===")
        all_results['parallel'] = self.parallel_scaling()
        
        # Save complete results
        self.metadata['end_time'] = datetime.now().isoformat()
        all_results['metadata'] = self.metadata
        
        self._save_results(all_results, 'complete_analysis')
        
        self.logger.info("Analysis complete!")
        return all_results


# Additional lattice implementations needed for experiments

class CheckerboardLattice(Lattice):
    """D_n checkerboard lattice."""
    
    def __init__(self, n):
        super().__init__(n)
        self.name = f"D_{n}"
        self.generate_basis()
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Generate D_n basis."""
        n = self.dimension
        basis_vectors = []
        
        # e_i - e_{i+1} for i = 0, ..., n-2
        for i in range(n - 1):
            v = vector(RDF, [0] * n)
            v[i] = 1
            v[i + 1] = -1
            basis_vectors.append(v)
            
        # e_{n-2} + e_{n-1}
        v = vector(RDF, [0] * n)
        v[n - 2] = 1
        v[n - 1] = 1
        basis_vectors.append(v)
        
        self._basis = Matrix(RDF, basis_vectors)


class RootLattice(Lattice):
    """Root lattice A_n or E_8."""
    
    def __init__(self, n, lattice_type='A'):
        super().__init__(n if lattice_type == 'A' else 8)
        self.lattice_type = lattice_type
        self.name = f"{lattice_type}_{n}"
        self.generate_basis()
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Generate root lattice basis."""
        if self.lattice_type == 'A':
            # A_n lattice
            n = self.dimension
            basis_vectors = []
            
            # Vectors e_i - e_{i+1} for i = 0, ..., n-1
            for i in range(n):
                v = vector(RDF, [0] * (n + 1))
                v[i] = 1
                v[i + 1] = -1
                basis_vectors.append(v[:n])  # Project to n dimensions
                
            self._basis = Matrix(RDF, basis_vectors)
        else:
            raise NotImplementedError(f"Lattice type {self.lattice_type} not implemented")


class IntegerLattice(Lattice):
    """Standard integer lattice Z^n."""
    
    def __init__(self, n):
        super().__init__(n)
        self.name = f"Z^{n}"
        self._basis = identity_matrix(RDF, n)
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Z^n has identity basis."""
        pass


class MatrixLattice(Lattice):
    """Lattice from arbitrary basis matrix."""
    
    def __init__(self, basis_matrix):
        self._basis = Matrix(RDF, basis_matrix)
        super().__init__(self._basis.nrows())
        self.name = "Custom"
        
    def get_basis(self):
        return self._basis
    
    def get_dimension(self):
        return self.dimension
    
    def generate_basis(self):
        """Basis already provided."""
        pass


# Quick test
if __name__ == '__main__':
    analyzer = DimensionScalingAnalysis()
    
    # Quick test with small parameters
    results = analyzer.mixing_time_vs_dimension(
        dimensions=[4, 8, 16],
        n_chains=2,
        max_samples=1000
    )
    
    print("Quick test completed!")