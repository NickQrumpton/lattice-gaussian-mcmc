#!/usr/bin/env python3
"""
Comprehensive performance benchmarking for lattice Gaussian sampling algorithms.

This script benchmarks sampling algorithms (Klein, IMHK) and reduction algorithms
(LLL, BKZ) across various lattices, dimensions, and parameters. Results are saved
in formats suitable for publication tables and performance analysis.

Usage:
    python benchmark_performance.py                  # Run all benchmarks
    python benchmark_performance.py --algorithms klein imhk  # Specific algorithms
    python benchmark_performance.py --dimensions 32 64 128   # Specific dimensions
"""

import numpy as np
import pandas as pd
import json
import time
import psutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.lattices.identity import IdentityLattice
from src.lattices.qary import QaryLattice
from src.lattices.ntru import NTRULattice
from src.lattices.reduction import LatticeReduction
from src.samplers.klein import KleinSampler
from src.samplers.imhk import IMHKSampler


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    # Algorithms to benchmark
    sampling_algorithms: List[str] = None  # ['klein', 'imhk']
    reduction_algorithms: List[str] = None  # ['lll', 'bkz_5', 'bkz_10', 'bkz_20']
    
    # Test lattices
    lattice_types: List[str] = None  # ['identity', 'qary', 'ntru']
    dimensions: List[int] = None  # [16, 32, 64, 128, 256]
    
    # Parameters
    sigma_factors: List[float] = None  # Multiples of smoothing parameter
    n_samples: int = 10000
    n_warmup: int = 100
    n_trials: int = 5
    
    # System settings
    n_cores: int = None
    memory_profiling: bool = False
    random_seed: int = 42
    
    # Output settings
    output_dir: str = "results/benchmarks"
    save_plots: bool = True
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if self.sampling_algorithms is None:
            self.sampling_algorithms = ['klein', 'imhk']
        if self.reduction_algorithms is None:
            self.reduction_algorithms = ['lll', 'bkz_5', 'bkz_10', 'bkz_20']
        if self.lattice_types is None:
            self.lattice_types = ['identity', 'qary', 'ntru']
        if self.dimensions is None:
            self.dimensions = [16, 32, 64, 128, 256]
        if self.sigma_factors is None:
            self.sigma_factors = [1.0, 2.0, 5.0]
        if self.n_cores is None:
            self.n_cores = mp.cpu_count()


@dataclass
class SamplingBenchmarkResult:
    """Results from sampling algorithm benchmark."""
    algorithm: str
    lattice_type: str
    dimension: int
    sigma: float
    sigma_over_eta: float
    time_per_sample: float
    time_std: float
    acceptance_rate: float
    ess_per_second: float
    memory_mb: float
    setup_time: float
    metadata: Dict[str, Any]


@dataclass
class ReductionBenchmarkResult:
    """Results from reduction algorithm benchmark."""
    algorithm: str
    lattice_type: str
    dimension: int
    reduction_time: float
    reduction_time_std: float
    hermite_factor: float
    orthogonality_defect: float
    memory_mb: float
    quality_metrics: Dict[str, float]


class PerformanceBenchmark:
    """Main class for performance benchmarking."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner."""
        self.config = config
        np.random.seed(config.random_seed)
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize reduction algorithms
        self.reducer = LatticeReduction()
        
        # Results storage
        self.sampling_results = []
        self.reduction_results = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file = self.output_dir / "benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        self.logger.info("Starting performance benchmarks...")
        
        # 1. Sampling benchmarks
        self.logger.info("Running sampling algorithm benchmarks...")
        self.run_sampling_benchmarks()
        
        # 2. Reduction benchmarks
        self.logger.info("Running reduction algorithm benchmarks...")
        self.run_reduction_benchmarks()
        
        # 3. Save results
        self.save_results()
        
        # 4. Generate reports
        self.generate_reports()
        
        self.logger.info("Benchmarks completed successfully!")
    
    def run_sampling_benchmarks(self):
        """Benchmark sampling algorithms."""
        # Test each combination of parameters
        for lattice_type in self.config.lattice_types:
            for dim in self.config.dimensions:
                # Create lattice
                lattice = self._create_lattice(lattice_type, dim)
                if lattice is None:
                    continue
                
                # Compute reference parameters
                eta = lattice.smoothing_parameter(epsilon=0.01)
                
                for sigma_factor in self.config.sigma_factors:
                    sigma = sigma_factor * eta
                    
                    self.logger.info(f"Benchmarking {lattice_type} n={dim} σ={sigma:.2f}")
                    
                    for algorithm in self.config.sampling_algorithms:
                        result = self._benchmark_sampling_algorithm(
                            algorithm, lattice, sigma
                        )
                        self.sampling_results.append(result)
    
    def run_reduction_benchmarks(self):
        """Benchmark reduction algorithms."""
        for lattice_type in self.config.lattice_types:
            for dim in self.config.dimensions:
                # Create lattice with random basis
                lattice = self._create_lattice(lattice_type, dim)
                if lattice is None:
                    continue
                
                # Get basis
                basis = lattice.get_basis()
                
                # Randomize basis for more realistic benchmark
                if lattice_type == 'identity':
                    # Apply random unimodular transformation
                    basis = self._randomize_basis(basis)
                
                self.logger.info(f"Benchmarking reduction for {lattice_type} n={dim}")
                
                for algorithm in self.config.reduction_algorithms:
                    result = self._benchmark_reduction_algorithm(
                        algorithm, basis, lattice_type
                    )
                    self.reduction_results.append(result)
    
    def _benchmark_sampling_algorithm(self, algorithm: str, lattice: Any, 
                                    sigma: float) -> SamplingBenchmarkResult:
        """Benchmark a single sampling algorithm."""
        # Memory usage before
        if self.config.memory_profiling:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            mem_before = 0
        
        # Setup sampler
        setup_start = time.time()
        if algorithm == 'klein':
            sampler = KleinSampler(lattice, sigma)
        elif algorithm == 'imhk':
            sampler = IMHKSampler(lattice, sigma)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        setup_time = time.time() - setup_start
        
        # Warmup
        for _ in range(self.config.n_warmup):
            if algorithm == 'klein':
                sampler.sample()
            else:
                sampler.step()
        
        # Benchmark over multiple trials
        trial_times = []
        
        for trial in range(self.config.n_trials):
            start_time = time.time()
            
            if algorithm == 'klein':
                # Direct sampling
                for _ in range(self.config.n_samples):
                    sampler.sample()
            else:
                # MCMC sampling
                samples = sampler.sample(
                    self.config.n_samples,
                    burn_in=0,  # Already warmed up
                    thin=1
                )
            
            elapsed = time.time() - start_time
            trial_times.append(elapsed)
        
        # Compute statistics
        time_per_sample = np.mean(trial_times) / self.config.n_samples
        time_std = np.std(trial_times) / self.config.n_samples
        
        # Algorithm-specific metrics
        if algorithm == 'klein':
            acceptance_rate = 1.0  # Always accepts
            ess_per_second = 1.0 / time_per_sample  # Each sample is independent
        else:
            acceptance_rate = sampler.acceptance_rate
            # Estimate ESS (simplified - assumes low autocorrelation)
            ess_per_second = acceptance_rate / time_per_sample
        
        # Memory usage after
        if self.config.memory_profiling:
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_mb = mem_after - mem_before
        else:
            memory_mb = 0
        
        # Compute sigma/eta ratio
        eta = lattice.smoothing_parameter(epsilon=0.01)
        sigma_over_eta = sigma / eta
        
        return SamplingBenchmarkResult(
            algorithm=algorithm,
            lattice_type=lattice.__class__.__name__,
            dimension=lattice.dimension,
            sigma=sigma,
            sigma_over_eta=sigma_over_eta,
            time_per_sample=time_per_sample,
            time_std=time_std,
            acceptance_rate=acceptance_rate,
            ess_per_second=ess_per_second,
            memory_mb=memory_mb,
            setup_time=setup_time,
            metadata={
                'n_samples': self.config.n_samples,
                'n_trials': self.config.n_trials
            }
        )
    
    def _benchmark_reduction_algorithm(self, algorithm: str, basis: np.ndarray,
                                     lattice_type: str) -> ReductionBenchmarkResult:
        """Benchmark a single reduction algorithm."""
        dim = basis.shape[0]
        
        # Memory usage before
        if self.config.memory_profiling:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
        else:
            mem_before = 0
        
        # Run reduction multiple times
        trial_times = []
        reduced_bases = []
        
        for trial in range(self.config.n_trials):
            basis_copy = basis.copy()
            
            start_time = time.time()
            
            if algorithm == 'lll':
                reduced_basis, stats = self.reducer.lll_reduce(basis_copy, delta=0.99)
            elif algorithm.startswith('bkz'):
                block_size = int(algorithm.split('_')[1])
                reduced_basis, stats = self.reducer.bkz_reduce(basis_copy, block_size=block_size)
            else:
                raise ValueError(f"Unknown reduction algorithm: {algorithm}")
            
            elapsed = time.time() - start_time
            trial_times.append(elapsed)
            reduced_bases.append((reduced_basis, stats))
        
        # Use the last result for quality metrics
        reduced_basis, stats = reduced_bases[-1]
        
        # Memory usage after
        if self.config.memory_profiling:
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_mb = mem_after - mem_before
        else:
            memory_mb = 0
        
        return ReductionBenchmarkResult(
            algorithm=algorithm,
            lattice_type=lattice_type,
            dimension=dim,
            reduction_time=np.mean(trial_times),
            reduction_time_std=np.std(trial_times),
            hermite_factor=stats.get('hermite_factor_after', 0),
            orthogonality_defect=stats.get('orthogonality_defect_after', 0),
            memory_mb=memory_mb,
            quality_metrics=stats
        )
    
    def _create_lattice(self, lattice_type: str, dimension: int) -> Optional[Any]:
        """Create test lattice."""
        try:
            if lattice_type == 'identity':
                return IdentityLattice(dimension)
            
            elif lattice_type == 'qary':
                m = 2 * dimension
                q = self._next_prime(10 * dimension)
                return QaryLattice.random_qary_lattice(dimension, m, q)
            
            elif lattice_type == 'ntru':
                if dimension & (dimension - 1) == 0 and dimension >= 16:
                    ntru_dim = dimension // 2
                    q = self._next_prime(20 * dimension)
                    lattice = NTRULattice(ntru_dim, q)
                    lattice.generate_basis()
                    return lattice
                else:
                    return None
            
        except Exception as e:
            self.logger.error(f"Failed to create {lattice_type} lattice: {e}")
            return None
    
    def _randomize_basis(self, basis: np.ndarray) -> np.ndarray:
        """Apply random unimodular transformation to basis."""
        n = basis.shape[0]
        
        # Generate random unimodular matrix
        U = np.eye(n, dtype=int)
        
        # Apply random row operations
        for _ in range(n):
            i, j = np.random.choice(n, 2, replace=False)
            k = np.random.randint(-3, 4)
            if k != 0:
                U[i] += k * U[j]
        
        # Apply transformation
        return U @ basis
    
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
        """Save benchmark results to files."""
        # Save sampling results
        if self.sampling_results:
            sampling_df = pd.DataFrame([asdict(r) for r in self.sampling_results])
            sampling_df.to_csv(self.output_dir / 'sampling_benchmarks.csv', index=False)
            
            # Save as JSON for nested data
            with open(self.output_dir / 'sampling_benchmarks.json', 'w') as f:
                json.dump([asdict(r) for r in self.sampling_results], f, indent=2)
        
        # Save reduction results
        if self.reduction_results:
            reduction_df = pd.DataFrame([asdict(r) for r in self.reduction_results])
            reduction_df.to_csv(self.output_dir / 'reduction_benchmarks.csv', index=False)
            
            with open(self.output_dir / 'reduction_benchmarks.json', 'w') as f:
                json.dump([asdict(r) for r in self.reduction_results], f, indent=2)
        
        # Save configuration
        with open(self.output_dir / 'benchmark_config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def generate_reports(self):
        """Generate performance reports and plots."""
        # Generate LaTeX tables
        self._generate_latex_tables()
        
        # Generate performance plots if enabled
        if self.config.save_plots:
            self._generate_performance_plots()
    
    def _generate_latex_tables(self):
        """Generate LaTeX-formatted tables."""
        # Table 1: Sampling performance
        if self.sampling_results:
            self._generate_sampling_table()
        
        # Table 2: Reduction performance
        if self.reduction_results:
            self._generate_reduction_table()
    
    def _generate_sampling_table(self):
        """Generate sampling performance table."""
        # Group results for summary
        df = pd.DataFrame([asdict(r) for r in self.sampling_results])
        
        # Summary by algorithm and dimension
        summary = df.groupby(['algorithm', 'dimension']).agg({
            'time_per_sample': ['mean', 'std'],
            'acceptance_rate': 'mean',
            'ess_per_second': 'mean'
        }).round(6)
        
        # Convert to LaTeX
        latex_table = summary.to_latex(
            caption='Sampling algorithm performance comparison',
            label='tab:sampling_performance',
            column_format='llrrrr'
        )
        
        with open(self.output_dir / 'sampling_performance.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_reduction_table(self):
        """Generate reduction performance table."""
        df = pd.DataFrame([asdict(r) for r in self.reduction_results])
        
        # Summary by algorithm and dimension
        summary = df.groupby(['algorithm', 'dimension']).agg({
            'reduction_time': ['mean', 'std'],
            'hermite_factor': 'mean',
            'orthogonality_defect': 'mean'
        }).round(3)
        
        # Convert to LaTeX
        latex_table = summary.to_latex(
            caption='Lattice reduction algorithm performance',
            label='tab:reduction_performance',
            column_format='llrrrr'
        )
        
        with open(self.output_dir / 'reduction_performance.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots."""
        plt.style.use('seaborn-v0_8-paper')
        
        # 1. Sampling time vs dimension
        self._plot_sampling_scaling()
        
        # 2. Reduction time vs dimension
        self._plot_reduction_scaling()
        
        # 3. Quality vs time trade-off
        self._plot_quality_tradeoff()
    
    def _plot_sampling_scaling(self):
        """Plot sampling time scaling with dimension."""
        if not self.sampling_results:
            return
        
        df = pd.DataFrame([asdict(r) for r in self.sampling_results])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time per sample
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            
            # Average over sigma values
            avg_by_dim = algo_data.groupby('dimension')['time_per_sample'].mean()
            
            ax1.loglog(avg_by_dim.index, avg_by_dim.values * 1000,
                      'o-', label=algo.upper(), markersize=8, linewidth=2)
        
        ax1.set_xlabel('Dimension n')
        ax1.set_ylabel('Time per Sample (ms)')
        ax1.set_title('Sampling Time Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ESS per second
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            avg_by_dim = algo_data.groupby('dimension')['ess_per_second'].mean()
            
            ax2.loglog(avg_by_dim.index, avg_by_dim.values,
                      's-', label=algo.upper(), markersize=8, linewidth=2)
        
        ax2.set_xlabel('Dimension n')
        ax2.set_ylabel('ESS per Second')
        ax2.set_title('Effective Sampling Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sampling_performance.pdf', dpi=300)
        plt.close()
    
    def _plot_reduction_scaling(self):
        """Plot reduction time scaling."""
        if not self.reduction_results:
            return
        
        df = pd.DataFrame([asdict(r) for r in self.reduction_results])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            avg_by_dim = algo_data.groupby('dimension')['reduction_time'].mean()
            
            ax.loglog(avg_by_dim.index, avg_by_dim.values,
                     'o-', label=algo.upper().replace('_', '-'),
                     markersize=8, linewidth=2)
        
        ax.set_xlabel('Dimension n')
        ax.set_ylabel('Reduction Time (s)')
        ax.set_title('Lattice Reduction Time Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reduction_performance.pdf', dpi=300)
        plt.close()
    
    def _plot_quality_tradeoff(self):
        """Plot quality vs time trade-off for reduction."""
        if not self.reduction_results:
            return
        
        df = pd.DataFrame([asdict(r) for r in self.reduction_results])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Group by algorithm
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            
            ax.scatter(algo_data['reduction_time'],
                      algo_data['hermite_factor'],
                      s=100, label=algo.upper().replace('_', '-'),
                      alpha=0.7)
        
        ax.set_xlabel('Reduction Time (s)')
        ax.set_ylabel('Hermite Factor')
        ax.set_title('Quality vs Time Trade-off')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_tradeoff.pdf', dpi=300)
        plt.close()


def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark lattice Gaussian sampling and reduction algorithms"
    )
    
    # Algorithm selection
    parser.add_argument(
        '--sampling-algorithms',
        nargs='+',
        choices=['klein', 'imhk'],
        help='Sampling algorithms to benchmark'
    )
    parser.add_argument(
        '--reduction-algorithms',
        nargs='+',
        choices=['lll', 'bkz_5', 'bkz_10', 'bkz_20'],
        help='Reduction algorithms to benchmark'
    )
    
    # Lattice parameters
    parser.add_argument(
        '--lattice-types',
        nargs='+',
        choices=['identity', 'qary', 'ntru'],
        help='Lattice types to test'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        help='Lattice dimensions to test'
    )
    
    # Benchmark parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of samples for benchmarking'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=5,
        help='Number of trials for each benchmark'
    )
    
    # System settings
    parser.add_argument(
        '--n-cores',
        type=int,
        help='Number of CPU cores to use'
    )
    parser.add_argument(
        '--memory-profiling',
        action='store_true',
        help='Enable memory profiling'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmarks',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        sampling_algorithms=args.sampling_algorithms,
        reduction_algorithms=args.reduction_algorithms,
        lattice_types=args.lattice_types,
        dimensions=args.dimensions,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        n_cores=args.n_cores,
        memory_profiling=args.memory_profiling,
        output_dir=args.output_dir,
        save_plots=not args.no_plots
    )
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(config)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()