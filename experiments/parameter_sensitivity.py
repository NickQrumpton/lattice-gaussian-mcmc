"""
Parameter Sensitivity Analysis for Lattice Gaussian MCMC.

This module implements comprehensive parameter sensitivity experiments to study
how key parameters (Ã, basis reduction, dimension, center vector) affect mixing
time, acceptance rate, and sample quality for Klein and IMHK samplers.

Experiments match or exceed the depth of analysis in Wang & Ling (2018).
"""

import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.lattices.base import Lattice
from src.lattices.identity import IdentityLattice
from src.lattices.qary import QaryLattice
from src.lattices.ntru import NTRULattice
from src.lattices.reduction import LatticeReduction
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IMHKSampler
from src.diagnostics.convergence import ConvergenceDiagnostics
from src.diagnostics.spectral import SpectralAnalysis
from src.samplers.utils import DiscreteGaussianUtils


@dataclass
class ExperimentConfig:
    """Configuration for parameter sensitivity experiments."""
    # Sigma parameters
    sigma_range_factors: List[float] = None  # Multiples of smoothing parameter
    n_sigma_points: int = 20
    
    # Basis reduction levels
    reduction_methods: List[str] = None  # ['none', 'lll', 'bkz', 'hkz']
    bkz_block_sizes: List[int] = None  # [2, 5, 10, 20]
    
    # Dimension parameters
    dimensions: List[int] = None  # [8, 16, 32, 64, 128]
    
    # Center vector parameters
    center_types: List[str] = None  # ['origin', 'random', 'deep_hole', 'boundary']
    n_center_samples: int = 10
    
    # Sampling parameters
    n_samples: int = 10000
    n_chains: int = 100
    burn_in: int = 1000
    thin: int = 10
    
    # Diagnostic parameters
    tvd_epsilon: float = 0.01
    spectral_gap_epsilon: float = 0.01
    autocorr_lags: int = 100
    
    # Computational parameters
    n_cores: int = None
    random_seed: int = 42
    
    # Output parameters
    output_dir: str = "results/parameter_sensitivity"
    save_raw_samples: bool = False
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.sigma_range_factors is None:
            self.sigma_range_factors = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        if self.reduction_methods is None:
            self.reduction_methods = ['none', 'lll', 'bkz_5', 'bkz_10', 'bkz_20']
        if self.dimensions is None:
            self.dimensions = [8, 16, 32, 64, 128]
        if self.center_types is None:
            self.center_types = ['origin', 'random', 'deep_hole', 'boundary']
        if self.bkz_block_sizes is None:
            self.bkz_block_sizes = [5, 10, 20]
        if self.n_cores is None:
            self.n_cores = mp.cpu_count()


@dataclass
class SamplerMetrics:
    """Metrics for evaluating sampler performance."""
    mixing_time: float
    acceptance_rate: float
    tvd_to_target: float
    spectral_gap: float
    autocorrelation_time: float
    ess_per_second: float  # Effective sample size per second
    max_gram_schmidt_norm: float
    condition_number: float
    computational_time: float


class ParameterSensitivityExperiments:
    """Main class for parameter sensitivity experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        np.random.seed(config.random_seed)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / "parameter_sensitivity.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize diagnostics
        self.convergence_diag = ConvergenceDiagnostics()
        self.spectral_analysis = SpectralAnalysis()
        self.dg_utils = DiscreteGaussianUtils()
        self.lattice_reducer = LatticeReduction()
        
        # Results storage
        self.results = {
            'sigma_sensitivity': {},
            'basis_sensitivity': {},
            'dimension_scaling': {},
            'center_sensitivity': {},
            'config': asdict(config)
        }
    
    def run_all_experiments(self):
        """Run complete parameter sensitivity analysis."""
        self.logger.info("Starting parameter sensitivity experiments...")
        
        # 1. Sigma sensitivity
        self.logger.info("Running sigma sensitivity analysis...")
        self.run_sigma_sensitivity()
        
        # 2. Basis quality sensitivity
        self.logger.info("Running basis reduction sensitivity analysis...")
        self.run_basis_sensitivity()
        
        # 3. Dimension scaling
        self.logger.info("Running dimension scaling analysis...")
        self.run_dimension_scaling()
        
        # 4. Center vector sensitivity
        self.logger.info("Running center vector sensitivity analysis...")
        self.run_center_sensitivity()
        
        # Save all results
        self.save_results()
        self.generate_summary_tables()
        
        self.logger.info("All experiments completed successfully!")
    
    def run_sigma_sensitivity(self):
        """Analyze sensitivity to Ã parameter."""
        results = []
        
        # Test on different lattice types
        test_lattices = self._get_test_lattices(dimension=64)
        
        for lattice_name, lattice in test_lattices.items():
            self.logger.info(f"  Testing sigma sensitivity on {lattice_name}")
            
            # Compute smoothing parameter and max GS norm
            eta = lattice.smoothing_parameter(self.config.spectral_gap_epsilon)
            gs_basis, _ = lattice.get_gram_schmidt()
            max_gs_norm = np.max(np.linalg.norm(gs_basis, axis=1))
            
            self.logger.info(f"    ·_µ(›) = {eta:.3f}, max||b*_i|| = {max_gs_norm:.3f}")
            
            # Generate sigma values
            sigma_values = self._generate_sigma_range(eta, max_gs_norm)
            
            # Test each sigma value
            for sigma in tqdm(sigma_values, desc=f"Sigma sweep for {lattice_name}"):
                # Run Klein sampler
                klein_metrics = self._evaluate_sampler(
                    lattice, 'klein', sigma, center=None
                )
                
                # Run IMHK sampler
                imhk_metrics = self._evaluate_sampler(
                    lattice, 'imhk', sigma, center=None
                )
                
                # Store results
                result = {
                    'lattice': lattice_name,
                    'dimension': lattice.dimension,
                    'sigma': sigma,
                    'sigma_over_eta': sigma / eta,
                    'sigma_over_max_gs': sigma / max_gs_norm,
                    'klein': asdict(klein_metrics),
                    'imhk': asdict(imhk_metrics),
                    'phase': self._detect_phase_transition(sigma, eta, max_gs_norm)
                }
                results.append(result)
        
        self.results['sigma_sensitivity'] = results
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'sigma_sensitivity.csv', index=False)
    
    def run_basis_sensitivity(self):
        """Analyze sensitivity to basis reduction."""
        results = []
        
        # Test lattices
        test_lattices = self._get_test_lattices(dimension=32)
        
        for lattice_name, base_lattice in test_lattices.items():
            self.logger.info(f"  Testing basis sensitivity on {lattice_name}")
            
            # Get original basis
            original_basis = base_lattice.get_basis()
            
            # Test different reduction methods
            for reduction_method in self.config.reduction_methods:
                self.logger.info(f"    Testing {reduction_method} reduction")
                
                # Apply reduction
                reduced_basis, reduction_stats = self._apply_reduction(
                    original_basis, reduction_method
                )
                
                # Create lattice with reduced basis
                reduced_lattice = self._create_lattice_with_basis(
                    base_lattice, reduced_basis
                )
                
                # Test multiple sigma values
                eta = reduced_lattice.smoothing_parameter(self.config.spectral_gap_epsilon)
                test_sigmas = [eta, 2*eta, 5*eta, 10*eta]
                
                for sigma in test_sigmas:
                    # Evaluate samplers
                    klein_metrics = self._evaluate_sampler(
                        reduced_lattice, 'klein', sigma
                    )
                    imhk_metrics = self._evaluate_sampler(
                        reduced_lattice, 'imhk', sigma
                    )
                    
                    # Store results
                    result = {
                        'lattice': lattice_name,
                        'dimension': reduced_lattice.dimension,
                        'reduction_method': reduction_method,
                        'sigma': sigma,
                        'sigma_over_eta': sigma / eta,
                        'reduction_stats': reduction_stats,
                        'klein': asdict(klein_metrics),
                        'imhk': asdict(imhk_metrics)
                    }
                    results.append(result)
        
        self.results['basis_sensitivity'] = results
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'basis_sensitivity.csv', index=False)
    
    def run_dimension_scaling(self):
        """Analyze how parameter effects scale with dimension."""
        results = []
        
        for dim in self.config.dimensions:
            self.logger.info(f"  Testing dimension n = {dim}")
            
            # Create test lattice
            lattice = IdentityLattice(dim)
            
            # Compute dimension-dependent parameters
            eta = lattice.smoothing_parameter(self.config.spectral_gap_epsilon)
            
            # Fixed sigma/sqrt(n) ratio
            sigma_over_sqrtn_values = [0.5, 1.0, 2.0, 5.0]
            
            for sigma_ratio in sigma_over_sqrtn_values:
                sigma = sigma_ratio * np.sqrt(dim)
                
                # Test with different reductions
                for reduction_method in ['none', 'lll', 'bkz_10']:
                    # Apply reduction
                    basis = lattice.get_basis()
                    reduced_basis, _ = self._apply_reduction(basis, reduction_method)
                    reduced_lattice = self._create_lattice_with_basis(
                        lattice, reduced_basis
                    )
                    
                    # Evaluate samplers
                    klein_metrics = self._evaluate_sampler(
                        reduced_lattice, 'klein', sigma
                    )
                    imhk_metrics = self._evaluate_sampler(
                        reduced_lattice, 'imhk', sigma
                    )
                    
                    # Store results
                    result = {
                        'dimension': dim,
                        'sigma': sigma,
                        'sigma_over_sqrtn': sigma_ratio,
                        'sigma_over_eta': sigma / eta,
                        'reduction_method': reduction_method,
                        'klein': asdict(klein_metrics),
                        'imhk': asdict(imhk_metrics)
                    }
                    results.append(result)
        
        self.results['dimension_scaling'] = results
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'dimension_scaling.csv', index=False)
    
    def run_center_sensitivity(self):
        """Analyze sensitivity to center vector choice."""
        results = []
        
        # Test on medium-sized lattices
        test_dimension = 32
        test_lattices = self._get_test_lattices(dimension=test_dimension)
        
        for lattice_name, lattice in test_lattices.items():
            self.logger.info(f"  Testing center sensitivity on {lattice_name}")
            
            # Fixed sigma values
            eta = lattice.smoothing_parameter(self.config.spectral_gap_epsilon)
            test_sigmas = [2*eta, 5*eta, 10*eta]
            
            for sigma in test_sigmas:
                # Test different center types
                for center_type in self.config.center_types:
                    # Generate centers
                    centers = self._generate_centers(
                        lattice, center_type, self.config.n_center_samples
                    )
                    
                    for i, center in enumerate(centers):
                        # Evaluate samplers
                        klein_metrics = self._evaluate_sampler(
                            lattice, 'klein', sigma, center=center
                        )
                        imhk_metrics = self._evaluate_sampler(
                            lattice, 'imhk', sigma, center=center
                        )
                        
                        # Compute center properties
                        center_norm = np.linalg.norm(center)
                        center_distance = self._distance_to_lattice(lattice, center)
                        
                        # Store results
                        result = {
                            'lattice': lattice_name,
                            'dimension': test_dimension,
                            'sigma': sigma,
                            'center_type': center_type,
                            'center_index': i,
                            'center_norm': center_norm,
                            'center_distance': center_distance,
                            'klein': asdict(klein_metrics),
                            'imhk': asdict(imhk_metrics)
                        }
                        results.append(result)
        
        self.results['center_sensitivity'] = results
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'center_sensitivity.csv', index=False)
    
    def _get_test_lattices(self, dimension: int) -> Dict[str, Lattice]:
        """Generate test lattices of given dimension."""
        lattices = {}
        
        # Identity lattice (Z^n)
        lattices['identity'] = IdentityLattice(dimension)
        
        # q-ary lattice
        if dimension <= 128:
            m = 2 * dimension
            q = self._next_prime(dimension * 10)
            lattices['qary'] = QaryLattice.random_qary_lattice(dimension, m, q)
        
        # NTRU lattice (if dimension is power of 2)
        if dimension & (dimension - 1) == 0 and dimension >= 16:
            lattices['ntru'] = NTRULattice(dimension // 2, q=self._next_prime(dimension * 20))
            lattices['ntru'].generate_basis()
        
        return lattices
    
    def _generate_sigma_range(self, eta: float, max_gs_norm: float) -> np.ndarray:
        """Generate range of sigma values for testing."""
        # Key transition points
        key_points = [
            0.5 * eta,    # Below smoothing parameter
            0.9 * eta,    # Just below eta
            eta,          # At smoothing parameter
            1.5 * eta,    # Above eta
            2 * eta,      # Well above eta
            0.5 * max_gs_norm,  # Half max GS norm
            max_gs_norm,        # At max GS norm
            2 * max_gs_norm,    # Above max GS norm
            5 * max_gs_norm,    # Well above max GS norm
        ]
        
        # Add intermediate points
        min_sigma = 0.5 * eta
        max_sigma = 10 * max_gs_norm
        
        # Log-spaced points for better coverage
        log_points = np.logspace(
            np.log10(min_sigma),
            np.log10(max_sigma),
            self.config.n_sigma_points
        )
        
        # Combine and sort
        all_points = np.unique(np.concatenate([key_points, log_points]))
        return all_points[all_points > 0]
    
    def _evaluate_sampler(self, lattice: Lattice, sampler_type: str, 
                         sigma: float, center: Optional[np.ndarray] = None) -> SamplerMetrics:
        """Evaluate sampler performance with given parameters."""
        start_time = time.time()
        
        # Create sampler
        if sampler_type == 'klein':
            sampler = KleinSampler(lattice, sigma)
        elif sampler_type == 'imhk':
            sampler = IMHKSampler(lattice, sigma)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Generate samples
        if sampler_type == 'klein':
            # Klein is direct sampler
            samples = np.array([
                sampler.sample(center=center) 
                for _ in range(self.config.n_samples)
            ])
            acceptance_rate = 1.0  # Always accepts
        else:
            # IMHK is MCMC
            if center is not None:
                sampler.current_state = center.copy()
            samples = sampler.sample(
                self.config.n_samples,
                burn_in=self.config.burn_in,
                thin=self.config.thin
            )
            acceptance_rate = sampler.acceptance_rate
        
        computational_time = time.time() - start_time
        
        # Compute metrics
        # 1. Mixing time (using multiple chains for MCMC)
        if sampler_type == 'imhk':
            mixing_time = self._estimate_mixing_time(
                lattice, sampler_type, sigma, center
            )
        else:
            mixing_time = 0.0  # Klein is direct
        
        # 2. TVD to target
        tvd = self._estimate_tvd(samples, lattice, sigma, center)
        
        # 3. Spectral gap
        if sampler_type == 'imhk':
            spectral_gap = sampler.spectral_gap()
        else:
            spectral_gap = 1.0  # Klein is direct
        
        # 4. Autocorrelation time
        autocorr_time = self._compute_autocorrelation_time(samples)
        
        # 5. Effective sample size per second
        ess = self.config.n_samples / (1 + 2 * autocorr_time)
        ess_per_second = ess / computational_time
        
        # 6. Lattice properties
        gs_basis, _ = lattice.get_gram_schmidt()
        max_gs_norm = np.max(np.linalg.norm(gs_basis, axis=1))
        condition_number = np.linalg.cond(lattice.get_basis())
        
        return SamplerMetrics(
            mixing_time=mixing_time,
            acceptance_rate=acceptance_rate,
            tvd_to_target=tvd,
            spectral_gap=spectral_gap,
            autocorrelation_time=autocorr_time,
            ess_per_second=ess_per_second,
            max_gram_schmidt_norm=max_gs_norm,
            condition_number=condition_number,
            computational_time=computational_time
        )
    
    def _estimate_mixing_time(self, lattice: Lattice, sampler_type: str,
                            sigma: float, center: Optional[np.ndarray]) -> float:
        """Estimate mixing time using multiple chains."""
        n_test_chains = min(20, self.config.n_chains)
        
        # Run chains from different starting points
        chain_samples = []
        for _ in range(n_test_chains):
            # Random initialization far from target
            if center is None:
                init_state = np.random.randn(lattice.dimension) * sigma * 10
            else:
                init_state = center + np.random.randn(lattice.dimension) * sigma * 10
            
            # Create sampler
            if sampler_type == 'imhk':
                sampler = IMHKSampler(lattice, sigma)
                sampler.current_state = init_state
                
                # Run chain
                samples = []
                for _ in range(self.config.n_samples // 10):
                    sampler.step()
                    samples.append(sampler.current_state.copy())
                
                chain_samples.append(samples)
        
        # Estimate mixing time from convergence diagnostics
        mixing_time = self.convergence_diag.estimate_mixing_time(
            chain_samples,
            epsilon=self.config.tvd_epsilon
        )
        
        return mixing_time
    
    def _estimate_tvd(self, samples: np.ndarray, lattice: Lattice,
                     sigma: float, center: Optional[np.ndarray]) -> float:
        """Estimate total variation distance to target distribution."""
        # For discrete Gaussian, compare empirical vs theoretical
        # This is approximate - true TVD computation is expensive
        
        # Compute empirical statistics
        if center is None:
            center = np.zeros(lattice.dimension)
        
        # Use coarse binning for TVD estimation
        # Project onto first few coordinates for efficiency
        n_proj = min(5, lattice.dimension)
        projected_samples = samples[:, :n_proj]
        projected_center = center[:n_proj]
        
        # Bin the samples
        n_bins = 20
        hist_range = [
            (projected_center[i] - 5*sigma, projected_center[i] + 5*sigma)
            for i in range(n_proj)
        ]
        
        empirical_hist, edges = np.histogramdd(
            projected_samples, bins=n_bins, range=hist_range
        )
        empirical_hist = empirical_hist / len(samples)
        
        # Compute theoretical probabilities (approximate)
        # This is simplified - full computation would use partition function
        theoretical_hist = self._compute_theoretical_histogram(
            edges, lattice, sigma, projected_center
        )
        
        # TVD = 0.5 * sum(|p - q|)
        tvd = 0.5 * np.sum(np.abs(empirical_hist - theoretical_hist))
        
        return tvd
    
    def _compute_theoretical_histogram(self, edges: List[np.ndarray], 
                                     lattice: Lattice, sigma: float,
                                     center: np.ndarray) -> np.ndarray:
        """Compute theoretical histogram for discrete Gaussian."""
        # Simplified computation - assumes independence for efficiency
        n_dims = len(edges) - 1
        shape = [len(e) - 1 for e in edges]
        hist = np.ones(shape)
        
        # For each dimension, compute 1D probabilities
        for i in range(n_dims):
            dim_edges = edges[i]
            dim_probs = []
            
            for j in range(len(dim_edges) - 1):
                # Probability mass in bin
                low, high = dim_edges[j], dim_edges[j+1]
                mid = (low + high) / 2
                
                # Approximate with continuous Gaussian
                prob = np.exp(-0.5 * (mid - center[i])**2 / sigma**2)
                dim_probs.append(prob)
            
            # Normalize
            dim_probs = np.array(dim_probs)
            dim_probs /= dim_probs.sum()
            
            # Update histogram
            for idx in np.ndindex(*shape):
                hist[idx] *= dim_probs[idx[i]]
        
        # Normalize
        hist /= hist.sum()
        return hist
    
    def _compute_autocorrelation_time(self, samples: np.ndarray) -> float:
        """Compute integrated autocorrelation time."""
        # Use first coordinate for efficiency
        x = samples[:, 0]
        
        # Compute autocorrelation function
        n = len(x)
        x_centered = x - np.mean(x)
        autocorr = np.correlate(x_centered, x_centered, mode='full')[n-1:]
        autocorr = autocorr / autocorr[0]
        
        # Integrate until first negative value or max lag
        tau = 0.5
        for i in range(1, min(self.config.autocorr_lags, n//4)):
            if autocorr[i] < 0:
                break
            tau += autocorr[i]
        
        return tau
    
    def _detect_phase_transition(self, sigma: float, eta: float, 
                               max_gs: float) -> str:
        """Detect phase transition regime."""
        if sigma < 0.9 * eta:
            return "below_smoothing"
        elif sigma < 1.1 * eta:
            return "near_smoothing"
        elif sigma < max_gs:
            return "intermediate"
        elif sigma < 2 * max_gs:
            return "near_klein_bound"
        else:
            return "large_sigma"
    
    def _apply_reduction(self, basis: np.ndarray, 
                        method: str) -> Tuple[np.ndarray, Dict]:
        """Apply specified basis reduction."""
        if method == 'none':
            return basis.copy(), {'method': 'none'}
        elif method == 'lll':
            return self.lattice_reducer.lll_reduce(basis, delta=0.99)
        elif method.startswith('bkz'):
            block_size = int(method.split('_')[1])
            return self.lattice_reducer.bkz_reduce(basis, block_size=block_size)
        elif method == 'hkz':
            # HKZ is exponential - only for small dimensions
            if basis.shape[0] <= 20:
                return self.lattice_reducer.hkz_reduce(basis)
            else:
                # Fall back to BKZ with large block size
                return self.lattice_reducer.bkz_reduce(basis, block_size=min(basis.shape[0], 30))
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def _create_lattice_with_basis(self, original_lattice: Lattice,
                                  new_basis: np.ndarray) -> Lattice:
        """Create lattice instance with new basis."""
        # This is a simplified approach - ideally each lattice class
        # would have a method to update its basis
        
        # For now, create a custom lattice wrapper
        class CustomLattice(Lattice):
            def __init__(self, basis):
                self.dimension = basis.shape[0]
                self._basis = basis
                self._gram_schmidt_computed = False
            
            def get_basis(self):
                return self._basis.copy()
            
            def decode_cvp(self, target):
                return self.nearest_plane(target)
        
        return CustomLattice(new_basis)
    
    def _generate_centers(self, lattice: Lattice, center_type: str,
                         n_samples: int) -> List[np.ndarray]:
        """Generate center vectors of specified type."""
        centers = []
        dim = lattice.dimension
        
        if center_type == 'origin':
            # Origin only
            centers = [np.zeros(dim) for _ in range(n_samples)]
            
        elif center_type == 'random':
            # Random points in fundamental region
            basis = lattice.get_basis()
            for _ in range(n_samples):
                # Random coefficients in [0, 1)
                coeffs = np.random.rand(dim)
                center = basis.T @ coeffs
                centers.append(center)
                
        elif center_type == 'deep_hole':
            # Deep holes (farthest from lattice points)
            # Approximate by points at half-integer coordinates
            basis = lattice.get_basis()
            for _ in range(n_samples):
                coeffs = np.random.rand(dim) * 0.5 + 0.25
                center = basis.T @ coeffs
                centers.append(center)
                
        elif center_type == 'boundary':
            # Points near covering radius
            basis = lattice.get_basis()
            covering_radius = self._estimate_covering_radius(lattice)
            
            for _ in range(n_samples):
                # Random direction
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction)
                
                # Scale to near covering radius
                scale = covering_radius * np.random.uniform(0.8, 1.0)
                center = direction * scale
                centers.append(center)
        
        return centers
    
    def _distance_to_lattice(self, lattice: Lattice, point: np.ndarray) -> float:
        """Compute distance from point to nearest lattice point."""
        nearest = lattice.decode_cvp(point)
        return np.linalg.norm(point - nearest)
    
    def _estimate_covering_radius(self, lattice: Lattice) -> float:
        """Estimate covering radius of lattice."""
        # Use Gaussian heuristic
        dim = lattice.dimension
        det = np.abs(np.linalg.det(lattice.get_basis()))
        
        # Covering radius H sqrt(dim/(2Àe)) * det^(1/dim)
        covering_radius = np.sqrt(dim / (2 * np.pi * np.e)) * det**(1/dim)
        
        return covering_radius
    
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
        # Save raw results as JSON
        results_file = self.output_dir / 'parameter_sensitivity_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as numpy archive
        np_file = self.output_dir / 'parameter_sensitivity_results.npz'
        np.savez_compressed(np_file, **self.results)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def generate_summary_tables(self):
        """Generate LaTeX-ready summary tables."""
        # Table 1: Sigma sensitivity summary
        self._generate_sigma_table()
        
        # Table 2: Basis reduction impact
        self._generate_basis_table()
        
        # Table 3: Dimension scaling summary
        self._generate_dimension_table()
        
        # Table 4: Center sensitivity summary
        self._generate_center_table()
    
    def _generate_sigma_table(self):
        """Generate LaTeX table for sigma sensitivity."""
        if 'sigma_sensitivity' not in self.results:
            return
        
        df = pd.DataFrame(self.results['sigma_sensitivity'])
        
        # Extract key metrics
        summary_data = []
        for lattice in df['lattice'].unique():
            lattice_df = df[df['lattice'] == lattice]
            
            for phase in ['below_smoothing', 'near_smoothing', 'intermediate', 'large_sigma']:
                phase_df = lattice_df[lattice_df['phase'] == phase]
                if len(phase_df) == 0:
                    continue
                
                # Average metrics in this phase
                klein_mixing = np.mean([r['mixing_time'] for r in phase_df['klein']])
                imhk_mixing = np.mean([r['mixing_time'] for r in phase_df['imhk']])
                imhk_accept = np.mean([r['acceptance_rate'] for r in phase_df['imhk']])
                
                summary_data.append({
                    'Lattice': lattice,
                    'Phase': phase.replace('_', ' '),
                    'Klein Mixing': f"{klein_mixing:.1f}",
                    'IMHK Mixing': f"{imhk_mixing:.1f}",
                    'IMHK Accept': f"{imhk_accept:.3f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption='Effect of Ã on sampler performance across phase transitions',
            label='tab:sigma_sensitivity',
            column_format='l|l|r|r|r'
        )
        
        with open(self.output_dir / 'sigma_sensitivity_table.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_basis_table(self):
        """Generate LaTeX table for basis reduction impact."""
        if 'basis_sensitivity' not in self.results:
            return
        
        df = pd.DataFrame(self.results['basis_sensitivity'])
        
        # Summary by reduction method
        summary_data = []
        for method in df['reduction_method'].unique():
            method_df = df[df['reduction_method'] == method]
            
            # Average improvement over unreduced
            if method != 'none':
                unreduced_df = df[df['reduction_method'] == 'none']
                
                # Match by lattice and sigma
                improvements = []
                for _, row in method_df.iterrows():
                    matching = unreduced_df[
                        (unreduced_df['lattice'] == row['lattice']) &
                        (unreduced_df['sigma'] == row['sigma'])
                    ]
                    if len(matching) > 0:
                        baseline = matching.iloc[0]
                        klein_speedup = baseline['klein']['computational_time'] / row['klein']['computational_time']
                        imhk_speedup = baseline['imhk']['mixing_time'] / row['imhk']['mixing_time']
                        improvements.append((klein_speedup, imhk_speedup))
                
                if improvements:
                    avg_klein_speedup = np.mean([x[0] for x in improvements])
                    avg_imhk_speedup = np.mean([x[1] for x in improvements])
                else:
                    avg_klein_speedup = 1.0
                    avg_imhk_speedup = 1.0
            else:
                avg_klein_speedup = 1.0
                avg_imhk_speedup = 1.0
            
            summary_data.append({
                'Reduction': method.replace('_', '-'),
                'Klein Speedup': f"{avg_klein_speedup:.2f}×",
                'IMHK Speedup': f"{avg_imhk_speedup:.2f}×"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption='Impact of basis reduction on sampling efficiency',
            label='tab:basis_reduction',
            column_format='l|r|r'
        )
        
        with open(self.output_dir / 'basis_reduction_table.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_dimension_table(self):
        """Generate LaTeX table for dimension scaling."""
        if 'dimension_scaling' not in self.results:
            return
        
        df = pd.DataFrame(self.results['dimension_scaling'])
        
        # Summary by dimension
        summary_data = []
        for dim in sorted(df['dimension'].unique()):
            dim_df = df[df['dimension'] == dim]
            
            # Average across different sigmas
            klein_time = np.mean([r['computational_time'] for r in dim_df['klein']])
            imhk_mixing = np.mean([r['mixing_time'] for r in dim_df['imhk']])
            imhk_gap = np.mean([r['spectral_gap'] for r in dim_df['imhk']])
            
            summary_data.append({
                'Dimension': dim,
                'Klein Time (s)': f"{klein_time:.3f}",
                'IMHK Mixing': f"{imhk_mixing:.0f}",
                'Spectral Gap': f"{imhk_gap:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption='Scaling of sampler performance with lattice dimension',
            label='tab:dimension_scaling',
            column_format='r|r|r|r'
        )
        
        with open(self.output_dir / 'dimension_scaling_table.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_center_table(self):
        """Generate LaTeX table for center sensitivity."""
        if 'center_sensitivity' not in self.results:
            return
        
        df = pd.DataFrame(self.results['center_sensitivity'])
        
        # Summary by center type
        summary_data = []
        for center_type in df['center_type'].unique():
            center_df = df[df['center_type'] == center_type]
            
            # Average metrics
            avg_distance = np.mean(center_df['center_distance'])
            klein_tvd = np.mean([r['tvd_to_target'] for r in center_df['klein']])
            imhk_tvd = np.mean([r['tvd_to_target'] for r in center_df['imhk']])
            
            summary_data.append({
                'Center Type': center_type.replace('_', ' '),
                'Avg Distance': f"{avg_distance:.3f}",
                'Klein TVD': f"{klein_tvd:.3f}",
                'IMHK TVD': f"{imhk_tvd:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption='Effect of center vector on sampling quality',
            label='tab:center_sensitivity',
            column_format='l|r|r|r'
        )
        
        with open(self.output_dir / 'center_sensitivity_table.tex', 'w') as f:
            f.write(latex_table)


def main():
    """Main entry point for parameter sensitivity experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis for lattice Gaussian MCMC"
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['sigma', 'basis', 'dimension', 'center', 'all'],
        default=['all'],
        help='Which sensitivity experiments to run'
    )
    
    # Parameter ranges
    parser.add_argument(
        '--sigma-range',
        nargs='+',
        type=float,
        help='Sigma range as multiples of smoothing parameter'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        help='Lattice dimensions to test'
    )
    parser.add_argument(
        '--reduction-methods',
        nargs='+',
        choices=['none', 'lll', 'bkz_5', 'bkz_10', 'bkz_20', 'hkz'],
        help='Basis reduction methods to test'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of samples per experiment'
    )
    parser.add_argument(
        '--n-chains',
        type=int,
        default=100,
        help='Number of chains for convergence diagnostics'
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
        help='Random seed for reproducibility'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/parameter_sensitivity',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-samples',
        action='store_true',
        help='Save raw samples (warning: large files)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        sigma_range_factors=args.sigma_range,
        dimensions=args.dimensions,
        reduction_methods=args.reduction_methods,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_cores=args.n_cores,
        random_seed=args.seed,
        output_dir=args.output_dir,
        save_raw_samples=args.save_samples
    )
    
    # Create experiment runner
    runner = ParameterSensitivityExperiments(config)
    
    # Run requested experiments
    if 'all' in args.experiments:
        runner.run_all_experiments()
    else:
        if 'sigma' in args.experiments:
            runner.run_sigma_sensitivity()
        if 'basis' in args.experiments:
            runner.run_basis_sensitivity()
        if 'dimension' in args.experiments:
            runner.run_dimension_scaling()
        if 'center' in args.experiments:
            runner.run_center_sensitivity()
        
        # Always save results and generate tables
        runner.save_results()
        runner.generate_summary_tables()


if __name__ == "__main__":
    main()