"""
Convergence Study for Lattice Gaussian Samplers.

This module implements comprehensive convergence analysis comparing Klein (direct)
and IMHK (Metropolis-Hastings Klein) samplers. It reproduces and extends the
convergence analysis from Wang & Ling (2018), particularly Figures 1 and 2.

Key analyses:
- TVD convergence curves for different algorithms
- Spectral gap estimation and scaling
- Dimension scaling behavior
- Mixing time diagnostics
"""

import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import eigh

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.lattices.base import Lattice
from src.lattices.identity import IdentityLattice
from src.lattices.qary import QaryLattice
from src.lattices.ntru import NTRULattice
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IMHKSampler
from src.diagnostics.convergence import ConvergenceDiagnostics
from src.diagnostics.spectral import SpectralAnalysis
from src.visualization.plots import PlottingTools


@dataclass
class ConvergenceConfig:
    """Configuration for convergence experiments."""
    # Test lattices
    lattice_types: List[str] = None  # ['identity', 'qary', 'ntru']
    dimensions: List[int] = None  # [8, 16, 32, 64, 128]
    
    # Sigma parameters
    sigma_factors: List[float] = None  # Multiples of smoothing parameter
    n_sigma_points: int = 10
    
    # Convergence parameters
    n_iterations: int = 10000
    n_chains: int = 100
    chain_burn_in: int = 0  # Start from cold start
    tvd_sample_interval: int = 10  # Compute TVD every N iterations
    
    # Ground truth parameters
    n_ground_truth_samples: int = 100000  # For Klein reference
    
    # Spectral gap parameters
    spectral_gap_methods: List[str] = None  # ['empirical', 'theoretical']
    spectral_gap_n_samples: int = 50000
    
    # Computational parameters
    n_cores: int = None
    random_seed: int = 42
    
    # Output parameters
    output_dir: str = "results/convergence_study"
    save_raw_chains: bool = False
    figure_format: str = "pdf"  # pdf, svg, or both
    figure_dpi: int = 300
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.lattice_types is None:
            self.lattice_types = ['identity', 'qary', 'ntru']
        if self.dimensions is None:
            self.dimensions = [8, 16, 32, 64, 128]
        if self.sigma_factors is None:
            self.sigma_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        if self.spectral_gap_methods is None:
            self.spectral_gap_methods = ['empirical', 'theoretical']
        if self.n_cores is None:
            self.n_cores = mp.cpu_count()


@dataclass
class ConvergenceResult:
    """Results from convergence experiment."""
    lattice_type: str
    dimension: int
    sigma: float
    sigma_over_eta: float
    algorithm: str
    iterations: np.ndarray
    tvd_values: np.ndarray
    tvd_mean: np.ndarray
    tvd_std: np.ndarray
    mixing_time: float
    spectral_gap: float
    acceptance_rate: float
    computational_time: float
    metadata: Dict[str, Any]


class ConvergenceStudy:
    """Main class for convergence experiments."""
    
    def __init__(self, config: ConvergenceConfig):
        """
        Initialize convergence study.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        np.random.seed(config.random_seed)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'figures': self.output_dir / 'figures',
            'data': self.output_dir / 'data',
            'tables': self.output_dir / 'tables',
            'logs': self.output_dir / 'logs'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.dirs['logs'] / "convergence_study.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize diagnostics and plotting
        self.convergence_diag = ConvergenceDiagnostics()
        self.spectral_analysis = SpectralAnalysis()
        self.plotter = PlottingTools()
        
        # Results storage
        self.results = {
            'convergence': [],
            'spectral_gaps': [],
            'dimension_scaling': [],
            'config': asdict(config)
        }
    
    def run_all_experiments(self):
        """Run complete convergence study."""
        self.logger.info("Starting convergence study experiments...")
        
        # 1. Algorithm comparison (Figure 1 style)
        self.logger.info("Running algorithm comparison experiments...")
        self.run_algorithm_comparison()
        
        # 2. Spectral gap analysis
        self.logger.info("Running spectral gap analysis...")
        self.run_spectral_gap_analysis()
        
        # 3. Dimension scaling (Figure 2 style)
        self.logger.info("Running dimension scaling experiments...")
        self.run_dimension_scaling()
        
        # Save all results
        self.save_results()
        
        # Generate all plots
        self.generate_plots()
        
        # Generate summary tables
        self.generate_tables()
        
        self.logger.info("Convergence study completed successfully!")
    
    def run_algorithm_comparison(self):
        """Compare Klein vs IMHK convergence for different settings."""
        # Test on medium dimension first
        test_dimension = 32
        
        for lattice_type in self.config.lattice_types:
            self.logger.info(f"Testing {lattice_type} lattice (n={test_dimension})")
            
            # Create lattice
            lattice = self._create_lattice(lattice_type, test_dimension)
            if lattice is None:
                continue
            
            # Compute reference parameters
            eta = lattice.smoothing_parameter(epsilon=0.01)
            gs_basis, _ = lattice.get_gram_schmidt()
            max_gs_norm = np.max(np.linalg.norm(gs_basis, axis=1))
            
            self.logger.info(f"  η_ε(Λ) = {eta:.3f}, max||b*_i|| = {max_gs_norm:.3f}")
            
            # Generate sigma values
            sigma_values = self._generate_sigma_values(eta, max_gs_norm)
            
            # Compute ground truth distribution for each sigma
            ground_truth_distributions = {}
            for sigma in sigma_values:
                self.logger.info(f"  Computing ground truth for σ = {sigma:.3f}")
                ground_truth = self._compute_ground_truth(lattice, sigma)
                ground_truth_distributions[sigma] = ground_truth
            
            # Run convergence experiments
            for sigma in sigma_values:
                self.logger.info(f"  Testing σ = {sigma:.3f} (σ/η = {sigma/eta:.2f})")
                
                # Klein convergence (instantaneous)
                klein_result = self._analyze_klein_convergence(
                    lattice, sigma, ground_truth_distributions[sigma]
                )
                self.results['convergence'].append(klein_result)
                
                # IMHK convergence
                imhk_result = self._analyze_imhk_convergence(
                    lattice, sigma, ground_truth_distributions[sigma]
                )
                self.results['convergence'].append(imhk_result)
    
    def run_spectral_gap_analysis(self):
        """Analyze spectral gap behavior for IMHK."""
        results = []
        
        # Test on different dimensions
        for dim in [16, 32, 64]:
            if dim not in self.config.dimensions:
                continue
            
            self.logger.info(f"Analyzing spectral gaps for dimension {dim}")
            
            # Test on identity lattice for consistency
            lattice = IdentityLattice(dim)
            eta = lattice.smoothing_parameter(epsilon=0.01)
            
            # Range of sigma values
            sigma_values = np.logspace(
                np.log10(0.5 * eta),
                np.log10(20 * eta),
                20
            )
            
            spectral_gaps = []
            theoretical_gaps = []
            
            for sigma in tqdm(sigma_values, desc=f"Spectral gap (n={dim})"):
                # Empirical spectral gap
                if 'empirical' in self.config.spectral_gap_methods:
                    emp_gap = self._estimate_empirical_spectral_gap(
                        lattice, sigma
                    )
                else:
                    emp_gap = np.nan
                
                # Theoretical spectral gap
                if 'theoretical' in self.config.spectral_gap_methods:
                    theo_gap = self._compute_theoretical_spectral_gap(
                        lattice, sigma
                    )
                else:
                    theo_gap = np.nan
                
                spectral_gaps.append(emp_gap)
                theoretical_gaps.append(theo_gap)
            
            result = {
                'dimension': dim,
                'sigma_values': sigma_values.tolist(),
                'sigma_over_eta': (sigma_values / eta).tolist(),
                'empirical_gaps': spectral_gaps,
                'theoretical_gaps': theoretical_gaps,
                'eta': eta
            }
            results.append(result)
        
        self.results['spectral_gaps'] = results
    
    def run_dimension_scaling(self):
        """Analyze how convergence scales with dimension."""
        results = []
        
        # Fixed sigma/sqrt(n) ratio
        sigma_over_sqrtn = 2.0
        
        for dim in self.config.dimensions:
            self.logger.info(f"Testing dimension scaling: n = {dim}")
            
            # Create lattice
            lattice = IdentityLattice(dim)
            sigma = sigma_over_sqrtn * np.sqrt(dim)
            
            # Compute ground truth
            ground_truth = self._compute_ground_truth(lattice, sigma)
            
            # Analyze IMHK convergence
            imhk_result = self._analyze_imhk_convergence(
                lattice, sigma, ground_truth,
                n_chains=min(self.config.n_chains, 50)  # Fewer chains for large dim
            )
            
            # Extract key metrics
            mixing_time = imhk_result.mixing_time
            spectral_gap = imhk_result.spectral_gap
            final_tvd = imhk_result.tvd_mean[-1]
            
            result = {
                'dimension': dim,
                'sigma': sigma,
                'sigma_over_sqrtn': sigma_over_sqrtn,
                'mixing_time': mixing_time,
                'spectral_gap': spectral_gap,
                'final_tvd': final_tvd,
                'iterations_to_01_tvd': self._find_convergence_time(
                    imhk_result.tvd_mean, threshold=0.01
                ),
                'iterations_to_001_tvd': self._find_convergence_time(
                    imhk_result.tvd_mean, threshold=0.001
                )
            }
            results.append(result)
        
        self.results['dimension_scaling'] = results
    
    def _create_lattice(self, lattice_type: str, dimension: int) -> Optional[Lattice]:
        """Create test lattice of specified type and dimension."""
        try:
            if lattice_type == 'identity':
                return IdentityLattice(dimension)
            
            elif lattice_type == 'qary':
                # q-ary lattice with reasonable parameters
                m = 2 * dimension
                q = self._next_prime(10 * dimension)
                return QaryLattice.random_qary_lattice(dimension, m, q)
            
            elif lattice_type == 'ntru':
                # NTRU only for power-of-2 dimensions
                if dimension & (dimension - 1) == 0 and dimension >= 16:
                    ntru_dim = dimension // 2
                    q = self._next_prime(20 * dimension)
                    lattice = NTRULattice(ntru_dim, q)
                    lattice.generate_basis()
                    return lattice
                else:
                    self.logger.warning(f"NTRU requires power-of-2 dimension, skipping n={dimension}")
                    return None
            
            else:
                raise ValueError(f"Unknown lattice type: {lattice_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to create {lattice_type} lattice: {e}")
            return None
    
    def _generate_sigma_values(self, eta: float, max_gs_norm: float) -> np.ndarray:
        """Generate sigma values for testing."""
        # Key transition points
        key_points = []
        
        # Around smoothing parameter
        for factor in self.config.sigma_factors:
            key_points.append(factor * eta)
        
        # Around max GS norm
        key_points.extend([0.5 * max_gs_norm, max_gs_norm, 2 * max_gs_norm])
        
        # Remove duplicates and sort
        sigma_values = np.unique(key_points)
        sigma_values = sigma_values[sigma_values > 0]
        
        return np.sort(sigma_values)
    
    def _compute_ground_truth(self, lattice: Lattice, sigma: float) -> Dict[str, Any]:
        """Compute ground truth distribution using Klein sampler."""
        self.logger.debug(f"Computing ground truth with {self.config.n_ground_truth_samples} samples")
        
        # Use Klein sampler for ground truth
        klein = KleinSampler(lattice, sigma)
        
        # Generate many samples
        samples = []
        for _ in tqdm(range(self.config.n_ground_truth_samples), 
                     desc="Ground truth", leave=False):
            samples.append(klein.sample())
        
        samples = np.array(samples)
        
        # Compute distribution statistics
        # For efficiency, project to lower dimension or use coarse binning
        ground_truth = {
            'samples': samples if self.config.save_raw_chains else None,
            'mean': np.mean(samples, axis=0),
            'cov': np.cov(samples.T),
            'second_moment': np.mean(np.sum(samples**2, axis=1)),
            'histogram': self._compute_histogram(samples, lattice.dimension)
        }
        
        return ground_truth
    
    def _compute_histogram(self, samples: np.ndarray, dim: int) -> Dict[str, Any]:
        """Compute histogram for TVD calculation."""
        # For high dimensions, use projection or coarse binning
        if dim > 10:
            # Project to first few coordinates
            n_proj = min(5, dim)
            samples_proj = samples[:, :n_proj]
        else:
            samples_proj = samples
        
        # Compute histogram
        n_bins = min(20, int(np.cbrt(len(samples))))
        hist, edges = np.histogramdd(samples_proj, bins=n_bins)
        hist = hist / len(samples)  # Normalize
        
        return {
            'hist': hist,
            'edges': edges,
            'n_proj': samples_proj.shape[1]
        }
    
    def _analyze_klein_convergence(self, lattice: Lattice, sigma: float,
                                  ground_truth: Dict[str, Any]) -> ConvergenceResult:
        """Analyze Klein sampler convergence (instantaneous)."""
        start_time = time.time()
        
        klein = KleinSampler(lattice, sigma)
        
        # Klein is direct sampler - "converges" immediately
        # We'll show this by computing TVD at different sample sizes
        sample_sizes = np.logspace(1, np.log10(self.config.n_iterations), 20).astype(int)
        tvd_values = []
        
        for n_samples in sample_sizes:
            # Generate samples
            samples = np.array([klein.sample() for _ in range(n_samples)])
            
            # Compute TVD
            tvd = self._compute_tvd(samples, ground_truth)
            tvd_values.append(tvd)
        
        # Klein has perfect acceptance and no mixing time
        result = ConvergenceResult(
            lattice_type=lattice.__class__.__name__,
            dimension=lattice.dimension,
            sigma=sigma,
            sigma_over_eta=sigma / lattice.smoothing_parameter(0.01),
            algorithm='klein',
            iterations=sample_sizes,
            tvd_values=np.array(tvd_values),
            tvd_mean=np.array(tvd_values),
            tvd_std=np.zeros_like(tvd_values),
            mixing_time=0.0,  # Direct sampler
            spectral_gap=1.0,  # Perfect mixing
            acceptance_rate=1.0,
            computational_time=time.time() - start_time,
            metadata={
                'direct_sampler': True,
                'sample_sizes': sample_sizes.tolist()
            }
        )
        
        return result
    
    def _analyze_imhk_convergence(self, lattice: Lattice, sigma: float,
                                 ground_truth: Dict[str, Any],
                                 n_chains: Optional[int] = None) -> ConvergenceResult:
        """Analyze IMHK sampler convergence using multiple chains."""
        start_time = time.time()
        
        if n_chains is None:
            n_chains = self.config.n_chains
        
        # Iterations to track
        track_iterations = np.arange(
            0, self.config.n_iterations + 1, 
            self.config.tvd_sample_interval
        )
        
        # Run multiple chains in parallel
        self.logger.debug(f"Running {n_chains} IMHK chains")
        
        # Use multiprocessing for parallel chains
        with mp.Pool(min(n_chains, self.config.n_cores)) as pool:
            chain_func = partial(
                self._run_single_imhk_chain,
                lattice=lattice,
                sigma=sigma,
                n_iterations=self.config.n_iterations,
                track_iterations=track_iterations,
                ground_truth=ground_truth
            )
            
            chain_results = list(tqdm(
                pool.imap(chain_func, range(n_chains)),
                total=n_chains,
                desc="IMHK chains"
            ))
        
        # Aggregate results
        all_tvd_curves = np.array([r['tvd_curve'] for r in chain_results])
        all_accept_rates = [r['acceptance_rate'] for r in chain_results]
        
        # Compute mean and std of TVD across chains
        tvd_mean = np.mean(all_tvd_curves, axis=0)
        tvd_std = np.std(all_tvd_curves, axis=0)
        
        # Estimate mixing time
        mixing_time = self._estimate_mixing_time_from_tvd(tvd_mean, threshold=0.01)
        
        # Estimate spectral gap
        imhk_sampler = IMHKSampler(lattice, sigma)
        spectral_gap = imhk_sampler.spectral_gap()
        
        result = ConvergenceResult(
            lattice_type=lattice.__class__.__name__,
            dimension=lattice.dimension,
            sigma=sigma,
            sigma_over_eta=sigma / lattice.smoothing_parameter(0.01),
            algorithm='imhk',
            iterations=track_iterations,
            tvd_values=all_tvd_curves,
            tvd_mean=tvd_mean,
            tvd_std=tvd_std,
            mixing_time=mixing_time,
            spectral_gap=spectral_gap,
            acceptance_rate=np.mean(all_accept_rates),
            computational_time=time.time() - start_time,
            metadata={
                'n_chains': n_chains,
                'theoretical_mixing_time': imhk_sampler.mixing_time(epsilon=0.01)
            }
        )
        
        return result
    
    def _run_single_imhk_chain(self, chain_id: int, lattice: Lattice, 
                              sigma: float, n_iterations: int,
                              track_iterations: np.ndarray,
                              ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single IMHK chain and track TVD."""
        # Set different random seed for each chain
        np.random.seed(self.config.random_seed + chain_id)
        
        # Create sampler with random initialization
        sampler = IMHKSampler(lattice, sigma)
        sampler.current_state = np.random.randn(lattice.dimension) * sigma * 5
        
        # Track TVD at specified iterations
        tvd_curve = []
        current_samples = []
        
        for i in range(n_iterations):
            # Take step
            sampler.step()
            current_samples.append(sampler.current_state.copy())
            
            # Compute TVD at tracking points
            if i + 1 in track_iterations:
                # Use recent samples for TVD
                recent_samples = np.array(current_samples[-1000:])
                tvd = self._compute_tvd(recent_samples, ground_truth)
                tvd_curve.append(tvd)
        
        return {
            'tvd_curve': tvd_curve,
            'acceptance_rate': sampler.acceptance_rate,
            'final_state': sampler.current_state
        }
    
    def _compute_tvd(self, samples: np.ndarray, ground_truth: Dict[str, Any]) -> float:
        """Compute total variation distance between samples and ground truth."""
        # Use histogram-based TVD for efficiency
        gt_hist_data = ground_truth['histogram']
        
        # Compute histogram with same binning
        if samples.shape[1] > gt_hist_data['n_proj']:
            samples_proj = samples[:, :gt_hist_data['n_proj']]
        else:
            samples_proj = samples
        
        # Use same edges as ground truth
        edges = gt_hist_data['edges']
        sample_hist = np.histogramdd(samples_proj, bins=edges)[0]
        sample_hist = sample_hist / len(samples)
        
        # TVD = 0.5 * sum(|p - q|)
        tvd = 0.5 * np.sum(np.abs(sample_hist.ravel() - gt_hist_data['hist'].ravel()))
        
        return tvd
    
    def _estimate_empirical_spectral_gap(self, lattice: Lattice, sigma: float) -> float:
        """Estimate spectral gap empirically from chain."""
        # Create sampler
        sampler = IMHKSampler(lattice, sigma)
        
        # Run chain and collect states
        n_samples = min(self.config.spectral_gap_n_samples, 10000)
        states = []
        
        for _ in range(n_samples):
            sampler.step()
            states.append(sampler.current_state.copy())
        
        states = np.array(states)
        
        # Estimate spectral gap from autocorrelation
        # Use first coordinate for efficiency
        x = states[:, 0]
        x_centered = x - np.mean(x)
        
        # Compute autocorrelation
        autocorr = np.correlate(x_centered, x_centered, mode='full')[len(x)-1:]
        autocorr = autocorr / autocorr[0]
        
        # Find first negative autocorrelation or exponential fit
        for lag in range(1, min(len(autocorr) // 4, 1000)):
            if autocorr[lag] < 0:
                # Estimate gap from exponential decay rate
                gap = -np.log(abs(autocorr[1])) if autocorr[1] > 0 else 1.0
                return min(gap, 1.0)
        
        # Fallback: fit exponential to first few lags
        lags = np.arange(1, min(10, len(autocorr)))
        log_autocorr = np.log(np.maximum(autocorr[lags], 1e-10))
        slope, _ = np.polyfit(lags, log_autocorr, 1)
        gap = -slope
        
        return min(max(gap, 0.0), 1.0)
    
    def _compute_theoretical_spectral_gap(self, lattice: Lattice, sigma: float) -> float:
        """Compute theoretical spectral gap bound."""
        # Use IMHK sampler's theoretical bound
        sampler = IMHKSampler(lattice, sigma)
        return sampler.spectral_gap()
    
    def _estimate_mixing_time_from_tvd(self, tvd_curve: np.ndarray, 
                                      threshold: float = 0.01) -> float:
        """Estimate mixing time from TVD curve."""
        # Find first iteration where TVD < threshold
        below_threshold = np.where(tvd_curve < threshold)[0]
        
        if len(below_threshold) > 0:
            # Return first iteration below threshold
            idx = below_threshold[0]
            if idx < len(self.config.n_iterations):
                return idx * self.config.tvd_sample_interval
        
        # If not converged, return total iterations
        return self.config.n_iterations
    
    def _find_convergence_time(self, tvd_curve: np.ndarray, threshold: float) -> float:
        """Find iteration where TVD first goes below threshold."""
        below = np.where(tvd_curve < threshold)[0]
        if len(below) > 0:
            return below[0] * self.config.tvd_sample_interval
        return np.inf
    
    def _next_prime(self, n: int) -> int:
        """Find next prime >= n."""
        candidate = n
        while True:
            if self._is_prime(candidate):
                return candidate
            candidate += 1
    
    def _is_prime(self, n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def save_results(self):
        """Save all experimental results."""
        # Save raw results
        results_file = self.dirs['data'] / 'convergence_results.json'
        
        # Convert numpy arrays to lists for JSON
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, list):
                json_results[key] = []
                for item in value:
                    if isinstance(item, ConvergenceResult):
                        # Convert dataclass to dict and handle numpy arrays
                        item_dict = asdict(item)
                        for k, v in item_dict.items():
                            if isinstance(v, np.ndarray):
                                item_dict[k] = v.tolist()
                        json_results[key].append(item_dict)
                    else:
                        json_results[key].append(item)
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save as numpy archive for easier loading
        np_file = self.dirs['data'] / 'convergence_results.npz'
        np.savez_compressed(np_file, **self.results)
        
        # Save convergence curves as CSV for easy plotting
        self._save_convergence_curves_csv()
        
        self.logger.info(f"Results saved to {self.dirs['data']}")
    
    def _save_convergence_curves_csv(self):
        """Save convergence curves in CSV format."""
        if 'convergence' not in self.results:
            return
        
        # Extract TVD curves
        records = []
        for result in self.results['convergence']:
            for i, iteration in enumerate(result.iterations):
                record = {
                    'lattice': result.lattice_type,
                    'dimension': result.dimension,
                    'sigma': result.sigma,
                    'sigma_over_eta': result.sigma_over_eta,
                    'algorithm': result.algorithm,
                    'iteration': iteration,
                    'tvd_mean': result.tvd_mean[i],
                    'tvd_std': result.tvd_std[i]
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(self.dirs['data'] / 'tvd_curves.csv', index=False)
    
    def generate_plots(self):
        """Generate all publication-quality plots."""
        self.logger.info("Generating plots...")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (8, 6),
            'figure.dpi': self.config.figure_dpi
        })
        
        # 1. TVD convergence comparison (Figure 1 style)
        self._plot_tvd_convergence()
        
        # 2. Spectral gap scaling (Figure 2 style)
        self._plot_spectral_gap_scaling()
        
        # 3. Dimension scaling
        self._plot_dimension_scaling()
        
        # 4. Algorithm comparison summary
        self._plot_algorithm_comparison()
    
    def _plot_tvd_convergence(self):
        """Plot TVD convergence curves (Figure 1 style)."""
        if 'convergence' not in self.results or not self.results['convergence']:
            return
        
        # Group by lattice type and sigma
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select representative sigma values
        results_df = pd.DataFrame([
            {
                'lattice': r.lattice_type,
                'sigma_over_eta': r.sigma_over_eta,
                'algorithm': r.algorithm,
                'result': r
            }
            for r in self.results['convergence']
        ])
        
        plot_idx = 0
        for lattice_type in ['IdentityLattice', 'QaryLattice', 'NTRULattice']:
            lattice_results = results_df[results_df['lattice'] == lattice_type]
            if len(lattice_results) == 0:
                continue
            
            # Select two sigma values: near eta and well above
            sigma_values = sorted(lattice_results['sigma_over_eta'].unique())
            selected_sigmas = []
            if len(sigma_values) >= 2:
                # One near 1.0, one around 5.0
                near_one = min(sigma_values, key=lambda x: abs(x - 1.0))
                near_five = min(sigma_values, key=lambda x: abs(x - 5.0))
                selected_sigmas = [near_one, near_five]
            else:
                selected_sigmas = sigma_values[:2]
            
            for sigma_ratio in selected_sigmas:
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                # Plot Klein and IMHK for this setting
                for algorithm in ['klein', 'imhk']:
                    result = lattice_results[
                        (lattice_results['sigma_over_eta'] == sigma_ratio) &
                        (lattice_results['algorithm'] == algorithm)
                    ]
                    
                    if len(result) > 0:
                        r = result.iloc[0]['result']
                        
                        # Plot TVD curve
                        if algorithm == 'klein':
                            # Klein converges immediately - show as horizontal line
                            ax.loglog(r.iterations, r.tvd_mean, 
                                     'o-', label='Klein', color='blue', markersize=4)
                        else:
                            # IMHK with confidence band
                            ax.loglog(r.iterations[1:], r.tvd_mean[1:], 
                                     '-', label='IMHK', color='red', linewidth=2)
                            ax.fill_between(
                                r.iterations[1:],
                                r.tvd_mean[1:] - r.tvd_std[1:],
                                r.tvd_mean[1:] + r.tvd_std[1:],
                                alpha=0.3, color='red'
                            )
                
                # Formatting
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Total Variation Distance')
                ax.set_title(f'{lattice_type.replace("Lattice", "")}, σ/η = {sigma_ratio:.1f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(1e-3, 1)
                
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save in requested formats
        for fmt in ['pdf', 'svg']:
            if fmt in self.config.figure_format or self.config.figure_format == 'both':
                plt.savefig(
                    self.dirs['figures'] / f'tvd_convergence.{fmt}',
                    format=fmt, bbox_inches='tight'
                )
        
        plt.close()
    
    def _plot_spectral_gap_scaling(self):
        """Plot spectral gap vs sigma (Figure 2 style)."""
        if 'spectral_gaps' not in self.results or not self.results['spectral_gaps']:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results['spectral_gaps'])))
        
        for i, result in enumerate(self.results['spectral_gaps']):
            dim = result['dimension']
            sigma_over_eta = np.array(result['sigma_over_eta'])
            emp_gaps = np.array(result['empirical_gaps'])
            theo_gaps = np.array(result['theoretical_gaps'])
            
            # Plot empirical gaps
            valid_emp = ~np.isnan(emp_gaps)
            if np.any(valid_emp):
                ax.semilogx(sigma_over_eta[valid_emp], emp_gaps[valid_emp],
                           'o-', color=colors[i], label=f'n={dim} (empirical)',
                           markersize=6)
            
            # Plot theoretical gaps
            valid_theo = ~np.isnan(theo_gaps)
            if np.any(valid_theo):
                ax.semilogx(sigma_over_eta[valid_theo], theo_gaps[valid_theo],
                           '--', color=colors[i], label=f'n={dim} (theoretical)',
                           linewidth=2)
        
        # Add reference lines
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='σ = η')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('σ/η')
        ax.set_ylabel('Spectral Gap')
        ax.set_title('Spectral Gap Scaling with σ')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['pdf', 'svg']:
            if fmt in self.config.figure_format or self.config.figure_format == 'both':
                plt.savefig(
                    self.dirs['figures'] / f'spectral_gap_scaling.{fmt}',
                    format=fmt, bbox_inches='tight'
                )
        
        plt.close()
    
    def _plot_dimension_scaling(self):
        """Plot how convergence scales with dimension."""
        if 'dimension_scaling' not in self.results or not self.results['dimension_scaling']:
            return
        
        df = pd.DataFrame(self.results['dimension_scaling'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mixing time vs dimension
        ax1.loglog(df['dimension'], df['mixing_time'], 'o-', markersize=8, linewidth=2)
        ax1.set_xlabel('Dimension n')
        ax1.set_ylabel('Mixing Time')
        ax1.set_title('Mixing Time Scaling')
        ax1.grid(True, alpha=0.3)
        
        # Add power law fit
        dims = df['dimension'].values
        mixing_times = df['mixing_time'].values
        valid = mixing_times > 0
        if np.sum(valid) > 2:
            log_dims = np.log(dims[valid])
            log_times = np.log(mixing_times[valid])
            slope, intercept = np.polyfit(log_dims, log_times, 1)
            fit_line = np.exp(intercept) * dims**slope
            ax1.loglog(dims, fit_line, '--', color='red', 
                      label=f'Slope = {slope:.2f}')
            ax1.legend()
        
        # Spectral gap vs dimension
        ax2.semilogx(df['dimension'], df['spectral_gap'], 's-', markersize=8, linewidth=2)
        ax2.set_xlabel('Dimension n')
        ax2.set_ylabel('Spectral Gap')
        ax2.set_title('Spectral Gap vs Dimension')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['pdf', 'svg']:
            if fmt in self.config.figure_format or self.config.figure_format == 'both':
                plt.savefig(
                    self.dirs['figures'] / f'dimension_scaling.{fmt}',
                    format=fmt, bbox_inches='tight'
                )
        
        plt.close()
    
    def _plot_algorithm_comparison(self):
        """Plot summary comparison of algorithms."""
        if 'convergence' not in self.results or not self.results['convergence']:
            return
        
        # Extract key metrics for comparison
        comparison_data = []
        
        for result in self.results['convergence']:
            comparison_data.append({
                'lattice': result.lattice_type.replace('Lattice', ''),
                'sigma_over_eta': result.sigma_over_eta,
                'algorithm': result.algorithm.upper(),
                'mixing_time': result.mixing_time,
                'final_tvd': result.tvd_mean[-1] if len(result.tvd_mean) > 0 else np.nan,
                'computational_time': result.computational_time
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Mixing time comparison
        ax = axes[0, 0]
        for lattice in df['lattice'].unique():
            lattice_df = df[df['lattice'] == lattice]
            for algo in ['KLEIN', 'IMHK']:
                algo_df = lattice_df[lattice_df['algorithm'] == algo]
                if len(algo_df) > 0:
                    ax.loglog(algo_df['sigma_over_eta'], algo_df['mixing_time'],
                             'o-' if algo == 'IMHK' else 's--',
                             label=f'{lattice} - {algo}')
        
        ax.set_xlabel('σ/η')
        ax.set_ylabel('Mixing Time')
        ax.set_title('Mixing Time Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Final TVD comparison
        ax = axes[0, 1]
        for algo in ['KLEIN', 'IMHK']:
            algo_df = df[df['algorithm'] == algo]
            if len(algo_df) > 0:
                ax.semilogy(algo_df['sigma_over_eta'], algo_df['final_tvd'],
                           'o-' if algo == 'IMHK' else 's--',
                           label=algo, markersize=8)
        
        ax.set_xlabel('σ/η')
        ax.set_ylabel('Final TVD')
        ax.set_title('Convergence Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Computational efficiency
        ax = axes[1, 0]
        klein_df = df[df['algorithm'] == 'KLEIN']
        imhk_df = df[df['algorithm'] == 'IMHK']
        
        if len(klein_df) > 0 and len(imhk_df) > 0:
            # Match by lattice and sigma
            speedup_data = []
            for _, klein_row in klein_df.iterrows():
                matching_imhk = imhk_df[
                    (imhk_df['lattice'] == klein_row['lattice']) &
                    (abs(imhk_df['sigma_over_eta'] - klein_row['sigma_over_eta']) < 0.1)
                ]
                if len(matching_imhk) > 0:
                    imhk_row = matching_imhk.iloc[0]
                    speedup = imhk_row['computational_time'] / klein_row['computational_time']
                    speedup_data.append({
                        'sigma_over_eta': klein_row['sigma_over_eta'],
                        'speedup': speedup,
                        'lattice': klein_row['lattice']
                    })
            
            speedup_df = pd.DataFrame(speedup_data)
            for lattice in speedup_df['lattice'].unique():
                lattice_speedup = speedup_df[speedup_df['lattice'] == lattice]
                ax.semilogx(lattice_speedup['sigma_over_eta'], 
                           lattice_speedup['speedup'],
                           'o-', label=lattice, markersize=8)
        
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('σ/η')
        ax.set_ylabel('Computational Time Ratio (IMHK/Klein)')
        ax.set_title('Computational Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        summary_text = "Algorithm Performance Summary\n\n"
        summary_text += "Klein:\n"
        summary_text += f"  - Direct sampling (no mixing time)\n"
        summary_text += f"  - Perfect acceptance rate\n"
        summary_text += f"  - Higher computational cost per sample\n\n"
        summary_text += "IMHK:\n"
        summary_text += f"  - Requires burn-in period\n"
        summary_text += f"  - Acceptance rate varies with σ\n"
        summary_text += f"  - More efficient for large batches\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save
        for fmt in ['pdf', 'svg']:
            if fmt in self.config.figure_format or self.config.figure_format == 'both':
                plt.savefig(
                    self.dirs['figures'] / f'algorithm_comparison.{fmt}',
                    format=fmt, bbox_inches='tight'
                )
        
        plt.close()
    
    def generate_tables(self):
        """Generate LaTeX tables for publication."""
        self.logger.info("Generating tables...")
        
        # Table 1: Convergence summary
        self._generate_convergence_table()
        
        # Table 2: Spectral gap summary
        self._generate_spectral_gap_table()
        
        # Table 3: Dimension scaling summary
        self._generate_dimension_scaling_table()
    
    def _generate_convergence_table(self):
        """Generate convergence summary table."""
        if 'convergence' not in self.results or not self.results['convergence']:
            return
        
        # Extract key metrics
        summary_data = []
        
        for result in self.results['convergence']:
            if result.algorithm == 'imhk':  # Focus on IMHK
                summary_data.append({
                    'Lattice': result.lattice_type.replace('Lattice', ''),
                    'n': result.dimension,
                    'σ/η': f"{result.sigma_over_eta:.1f}",
                    'Mixing Time': f"{result.mixing_time:.0f}",
                    'Accept Rate': f"{result.acceptance_rate:.3f}",
                    'Spectral Gap': f"{result.spectral_gap:.3f}"
                })
        
        df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = df.to_latex(
            index=False,
            caption='IMHK convergence properties for different lattices and parameters',
            label='tab:convergence_summary',
            column_format='l|r|r|r|r|r'
        )
        
        with open(self.dirs['tables'] / 'convergence_summary.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_spectral_gap_table(self):
        """Generate spectral gap summary table."""
        if 'spectral_gaps' not in self.results or not self.results['spectral_gaps']:
            return
        
        # Extract key points
        summary_data = []
        
        for result in self.results['spectral_gaps']:
            dim = result['dimension']
            sigma_values = np.array(result['sigma_over_eta'])
            emp_gaps = np.array(result['empirical_gaps'])
            theo_gaps = np.array(result['theoretical_gaps'])
            
            # Find gap at key sigma values
            for sigma_target in [1.0, 2.0, 5.0, 10.0]:
                idx = np.argmin(np.abs(sigma_values - sigma_target))
                if idx < len(sigma_values):
                    summary_data.append({
                        'n': dim,
                        'σ/η': f"{sigma_values[idx]:.1f}",
                        'Empirical Gap': f"{emp_gaps[idx]:.3f}" if not np.isnan(emp_gaps[idx]) else "—",
                        'Theoretical Gap': f"{theo_gaps[idx]:.3f}" if not np.isnan(theo_gaps[idx]) else "—"
                    })
        
        df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = df.to_latex(
            index=False,
            caption='Spectral gap estimates for identity lattice',
            label='tab:spectral_gaps',
            column_format='r|r|r|r'
        )
        
        with open(self.dirs['tables'] / 'spectral_gaps.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_dimension_scaling_table(self):
        """Generate dimension scaling summary table."""
        if 'dimension_scaling' not in self.results or not self.results['dimension_scaling']:
            return
        
        df = pd.DataFrame(self.results['dimension_scaling'])
        
        # Format for table
        summary_df = df[['dimension', 'mixing_time', 'spectral_gap', 
                        'iterations_to_01_tvd', 'iterations_to_001_tvd']].copy()
        
        summary_df.columns = ['n', 'Mixing Time', 'Spectral Gap', 
                             'Iter to 0.01 TVD', 'Iter to 0.001 TVD']
        
        # Format numbers
        summary_df['Mixing Time'] = summary_df['Mixing Time'].apply(lambda x: f"{x:.0f}")
        summary_df['Spectral Gap'] = summary_df['Spectral Gap'].apply(lambda x: f"{x:.3f}")
        summary_df['Iter to 0.01 TVD'] = summary_df['Iter to 0.01 TVD'].apply(
            lambda x: f"{x:.0f}" if x < np.inf else "—"
        )
        summary_df['Iter to 0.001 TVD'] = summary_df['Iter to 0.001 TVD'].apply(
            lambda x: f"{x:.0f}" if x < np.inf else "—"
        )
        
        # Generate LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption='Dimension scaling of IMHK convergence (σ = 2√n)',
            label='tab:dimension_scaling',
            column_format='r|r|r|r|r'
        )
        
        with open(self.dirs['tables'] / 'dimension_scaling.tex', 'w') as f:
            f.write(latex_table)


def main():
    """Main entry point for convergence study."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run convergence study for lattice Gaussian samplers"
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['comparison', 'spectral', 'scaling', 'all'],
        default=['all'],
        help='Which experiments to run'
    )
    
    # Lattice parameters
    parser.add_argument(
        '--lattice-types',
        nargs='+',
        choices=['identity', 'qary', 'ntru'],
        default=['identity', 'qary', 'ntru'],
        help='Lattice types to test'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        default=[8, 16, 32, 64],
        help='Lattice dimensions to test'
    )
    
    # Convergence parameters
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=10000,
        help='Number of iterations for convergence'
    )
    parser.add_argument(
        '--n-chains',
        type=int,
        default=100,
        help='Number of parallel chains for IMHK'
    )
    parser.add_argument(
        '--n-ground-truth',
        type=int,
        default=100000,
        help='Number of samples for ground truth'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/convergence_study',
        help='Output directory'
    )
    parser.add_argument(
        '--figure-format',
        choices=['pdf', 'svg', 'both'],
        default='pdf',
        help='Figure output format'
    )
    parser.add_argument(
        '--save-chains',
        action='store_true',
        help='Save raw chain data (warning: large files)'
    )
    
    # Computational parameters
    parser.add_argument(
        '--n-cores',
        type=int,
        help='Number of CPU cores to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ConvergenceConfig(
        lattice_types=args.lattice_types,
        dimensions=args.dimensions,
        n_iterations=args.n_iterations,
        n_chains=args.n_chains,
        n_ground_truth_samples=args.n_ground_truth,
        output_dir=args.output_dir,
        figure_format=args.figure_format,
        save_raw_chains=args.save_chains,
        n_cores=args.n_cores,
        random_seed=args.seed
    )
    
    # Create and run study
    study = ConvergenceStudy(config)
    
    if 'all' in args.experiments:
        study.run_all_experiments()
    else:
        if 'comparison' in args.experiments:
            study.run_algorithm_comparison()
        if 'spectral' in args.experiments:
            study.run_spectral_gap_analysis()
        if 'scaling' in args.experiments:
            study.run_dimension_scaling()
        
        # Always save and plot
        study.save_results()
        study.generate_plots()
        study.generate_tables()


if __name__ == "__main__":
    main()