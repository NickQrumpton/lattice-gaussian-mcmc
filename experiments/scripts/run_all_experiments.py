#!/usr/bin/env python3
"""
Master script to run all experiments for the lattice Gaussian MCMC paper.

This script orchestrates all experiments needed to reproduce the results
in Wang & Ling (2018) and generate publication-ready figures and tables.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.cryptographic_experiments import CryptographicExperiments
from experiments.dimension_scaling import DimensionScalingExperiments
from experiments.convergence_study import ConvergenceStudy
from src.visualization.plots import PlottingTools


class ExperimentRunner:
    """Master experiment runner for all paper results."""
    
    def __init__(self, output_dir: str = "results", n_cores: int = None):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            n_cores: Number of CPU cores to use (None for all)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'figures': self.output_dir / 'figures',
            'tables': self.output_dir / 'tables',
            'data': self.output_dir / 'data',
            'logs': self.output_dir / 'logs'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.dirs['logs'] / f"experiment_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # CPU cores
        self.n_cores = n_cores or mp.cpu_count()
        self.logger.info(f"Using {self.n_cores} CPU cores")
        
        # Initialize experiment classes
        self.crypto_exp = CryptographicExperiments()
        self.scaling_exp = DimensionScalingExperiments()
        self.convergence_exp = ConvergenceStudy()
        self.plotter = PlottingTools()
        
        # Results storage
        self.results = {}
    
    def run_all_experiments(self):
        """Run all experiments sequentially."""
        self.logger.info("Starting all experiments...")
        
        # 1. Convergence comparison (Figure 1 in paper)
        self.logger.info("Running convergence comparison experiments...")
        self.run_convergence_comparison()
        
        # 2. Dimension scaling analysis (Figure 2 in paper)
        self.logger.info("Running dimension scaling experiments...")
        self.run_dimension_scaling()
        
        # 3. Cryptographic lattice experiments (Table 1 in paper)
        self.logger.info("Running cryptographic lattice experiments...")
        self.run_cryptographic_experiments()
        
        # 4. Parameter sensitivity analysis (Figure 3 in paper)
        self.logger.info("Running parameter sensitivity experiments...")
        self.run_parameter_sensitivity()
        
        # 5. Spectral gap analysis (Figure 4 in paper)
        self.logger.info("Running spectral gap experiments...")
        self.run_spectral_gap_analysis()
        
        # Save all results
        self.save_all_results()
        
        # Generate all figures and tables
        self.generate_publication_outputs()
        
        self.logger.info("All experiments completed successfully!")
    
    def run_convergence_comparison(self):
        """Compare Klein vs IMHK convergence."""
        dimensions = [50, 100, 200]
        sigma_values = [1.0, 5.0, 10.0]
        
        results = {}
        for dim in dimensions:
            for sigma in sigma_values:
                self.logger.info(f"  Convergence: dim={dim}, sigma={sigma}")
                
                # Run experiment
                tvd_klein, tvd_imhk, times = self.convergence_exp.compare_convergence(
                    dimension=dim,
                    sigma=sigma,
                    n_steps=10000,
                    n_chains=100
                )
                
                results[f"dim{dim}_sigma{sigma}"] = {
                    'tvd_klein': tvd_klein,
                    'tvd_imhk': tvd_imhk,
                    'times': times,
                    'dimension': dim,
                    'sigma': sigma
                }
        
        self.results['convergence_comparison'] = results
        
        # Generate figure
        self.plotter.plot_convergence_comparison(
            results,
            save_path=self.dirs['figures'] / 'convergence_comparison.pdf'
        )
    
    def run_dimension_scaling(self):
        """Analyze scaling with dimension."""
        dimensions = np.arange(10, 510, 50)
        sigma_over_sqrt_n = 2.0  # σ/√n fixed
        
        results = {
            'dimensions': dimensions,
            'mixing_times': [],
            'spectral_gaps': [],
            'computational_times': [],
            'delta_inverse_values': []
        }
        
        for dim in tqdm(dimensions, desc="Dimension scaling"):
            sigma = sigma_over_sqrt_n * np.sqrt(dim)
            
            # Run scaling analysis
            mixing_time, spectral_gap, comp_time = self.scaling_exp.analyze_dimension(
                dimension=dim,
                sigma=sigma,
                n_samples=5000
            )
            
            results['mixing_times'].append(mixing_time)
            results['spectral_gaps'].append(spectral_gap)
            results['computational_times'].append(comp_time)
            
            # Compute δ^{-1}
            delta_inv = self.scaling_exp.compute_delta_inverse(dim, sigma)
            results['delta_inverse_values'].append(delta_inv)
        
        self.results['dimension_scaling'] = results
        
        # Generate figures
        self.plotter.plot_dimension_scaling(
            results,
            save_path=self.dirs['figures'] / 'dimension_scaling.pdf'
        )
    
    def run_cryptographic_experiments(self):
        """Run experiments on cryptographic lattices."""
        # Lattice parameters from Table 1 in paper
        lattice_configs = [
            {'name': 'NTRU-256', 'n': 256, 'q': 2048, 'sigma': 4.0},
            {'name': 'NTRU-512', 'n': 512, 'q': 4096, 'sigma': 8.0},
            {'name': 'Falcon-512', 'n': 512, 'q': 12289, 'sigma': 10.0},
            {'name': 'Falcon-1024', 'n': 1024, 'q': 12289, 'sigma': 14.0},
        ]
        
        results = []
        for config in lattice_configs:
            self.logger.info(f"  Testing {config['name']}...")
            
            # Run experiment
            result = self.crypto_exp.benchmark_lattice(
                lattice_type='ntru',
                dimension=config['n'],
                q=config['q'],
                sigma=config['sigma'],
                n_samples=10000
            )
            
            result['name'] = config['name']
            results.append(result)
        
        self.results['cryptographic'] = results
        
        # Generate table
        self.generate_crypto_table(results)
    
    def run_parameter_sensitivity(self):
        """Analyze sensitivity to σ parameter."""
        dimension = 256
        sigma_base = 5.0
        sigma_range = np.linspace(0.5, 2.0, 20) * sigma_base
        
        results = {
            'sigma_values': sigma_range,
            'mixing_times': [],
            'spectral_gaps': [],
            'acceptance_rates': []
        }
        
        for sigma in tqdm(sigma_range, desc="Parameter sensitivity"):
            # Run sensitivity analysis
            mixing_time, spectral_gap, accept_rate = self.crypto_exp.sigma_sensitivity(
                dimension=dimension,
                sigma=sigma,
                n_samples=5000
            )
            
            results['mixing_times'].append(mixing_time)
            results['spectral_gaps'].append(spectral_gap)
            results['acceptance_rates'].append(accept_rate)
        
        self.results['parameter_sensitivity'] = results
        
        # Generate figure
        self.plotter.plot_parameter_sensitivity(
            results,
            save_path=self.dirs['figures'] / 'parameter_sensitivity.pdf'
        )
    
    def run_spectral_gap_analysis(self):
        """Analyze spectral gap behavior."""
        dimensions = [50, 100, 200]
        sigma_over_eta_range = np.logspace(-0.5, 1.5, 20)
        
        results = {}
        for dim in dimensions:
            gaps = []
            for sigma_ratio in sigma_over_eta_range:
                # Compute spectral gap
                gap = self.convergence_exp.compute_spectral_gap_ratio(
                    dimension=dim,
                    sigma_over_eta=sigma_ratio
                )
                gaps.append(gap)
            
            results[f'dim_{dim}'] = {
                'sigma_over_eta': sigma_over_eta_range,
                'spectral_gaps': gaps
            }
        
        self.results['spectral_gap'] = results
        
        # Generate figure
        self.plotter.plot_spectral_gap_analysis(
            results,
            save_path=self.dirs['figures'] / 'spectral_gap_analysis.pdf'
        )
    
    def generate_crypto_table(self, results):
        """Generate LaTeX table for cryptographic results."""
        df = pd.DataFrame(results)
        
        # Format table
        latex_table = df.to_latex(
            index=False,
            columns=['name', 'klein_time', 'imhk_time', 'speedup', 
                    'klein_tvd', 'imhk_tvd', 'spectral_gap'],
            column_format='l|rr|r|rr|r',
            float_format='%.3f',
            caption='Performance comparison on cryptographic lattices',
            label='tab:crypto_results'
        )
        
        # Save table
        with open(self.dirs['tables'] / 'cryptographic_results.tex', 'w') as f:
            f.write(latex_table)
    
    def save_all_results(self):
        """Save all experimental results to disk."""
        # Save as JSON
        results_file = self.dirs['data'] / 'all_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        # Save as numpy archive
        np_file = self.dirs['data'] / 'all_results.npz'
        np.savez_compressed(np_file, **self.results)
        
        self.logger.info(f"Results saved to {results_file} and {np_file}")
    
    def generate_publication_outputs(self):
        """Generate all figures and tables for publication."""
        self.logger.info("Generating publication outputs...")
        
        # Copy figures to paper directory
        paper_fig_dir = Path("paper/figures")
        if paper_fig_dir.exists():
            import shutil
            for fig_file in self.dirs['figures'].glob("*.pdf"):
                shutil.copy2(fig_file, paper_fig_dir / fig_file.name)
        
        # Copy tables to paper directory
        paper_tab_dir = Path("paper/tables")
        if paper_tab_dir.exists():
            import shutil
            for tab_file in self.dirs['tables'].glob("*.tex"):
                shutil.copy2(tab_file, paper_tab_dir / tab_file.name)
        
        self.logger.info("Publication outputs generated!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all experiments for lattice Gaussian MCMC paper"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--n-cores", 
        type=int, 
        default=None,
        help="Number of CPU cores to use (default: all)"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=['convergence', 'scaling', 'crypto', 'sensitivity', 'spectral', 'all'],
        default=['all'],
        help="Which experiments to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        n_cores=args.n_cores
    )
    
    # Run requested experiments
    if 'all' in args.experiments:
        runner.run_all_experiments()
    else:
        if 'convergence' in args.experiments:
            runner.run_convergence_comparison()
        if 'scaling' in args.experiments:
            runner.run_dimension_scaling()
        if 'crypto' in args.experiments:
            runner.run_cryptographic_experiments()
        if 'sensitivity' in args.experiments:
            runner.run_parameter_sensitivity()
        if 'spectral' in args.experiments:
            runner.run_spectral_gap_analysis()
        
        # Save results and generate outputs
        runner.save_all_results()
        runner.generate_publication_outputs()


if __name__ == "__main__":
    main()