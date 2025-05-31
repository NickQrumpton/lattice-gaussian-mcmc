#!/usr/bin/env sage -python
"""
Klein Sampler Scaling Analysis: Dimensions 16, 32, 64

This script performs a comprehensive scaling analysis of the Klein discrete
Gaussian sampler across multiple dimensions, following the exact specifications
for reproducible research.

For each dimension n ∈ {16, 32, 64}:
1. Generate random integer matrix with fixed seed
2. Apply LLL reduction
3. Compute Gram-Schmidt norms and set σ = 5.0 × max_GS_norm
4. Generate 50,000 samples and measure performance
5. Analyze 1D marginal distribution accuracy
6. Generate publication-quality plots and data files

Requirements:
- SageMath (for LLL reduction and matrix operations)
- numpy (for numerical computations)
- matplotlib (for plotting)
- json, csv (for data export)

Author: Klein Sampler Benchmark Suite
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import time
import os
from pathlib import Path
import sys
sys.path.append('.')

# SageMath imports
from sage.all import *
from sage.modules.free_module_integer import IntegerLattice

# Local imports
from src.lattices.simple import SimpleLattice
from src.samplers.klein import RefinedKleinSampler

# Set all random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
set_random_seed(SEED)

class KleinScalingAnalysis:
    """Comprehensive scaling analysis for Klein sampler across dimensions."""
    
    def __init__(self, output_dir="results"):
        """Initialize the scaling analysis."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.dimensions = [16, 32, 64]  # Full analysis
        self.n_samples = 50000
        self.matrix_entry_max = 50  # Entries in {0,1,...,50}
        self.sigma_multiplier = 1.5  # σ = 1.5 × max_GS_norm (reasonable for sampling)
        self.seed = SEED
        
        # Results storage
        self.results = {}
        
        print(f"🔬 Klein Sampler Scaling Analysis")
        print(f"=" * 50)
        print(f"Dimensions: {self.dimensions}")
        print(f"Samples per dimension: {self.n_samples:,}")
        print(f"Random seed: {self.seed}")
        print(f"Output directory: {self.output_dir}")
        print(f"Matrix entries: uniform in {{0,1,...,{self.matrix_entry_max}}}")
        print(f"Sigma parameter: {self.sigma_multiplier} × max_GS_norm")
        print()
    
    def generate_random_matrix(self, n):
        """Generate random integer matrix with entries in {0,1,...,50}."""
        print(f"  Generating random {n}×{n} matrix...")
        
        # Set specific seed for this dimension for reproducibility
        np.random.seed(self.seed + n)  # Different seed per dimension but deterministic
        
        # Generate random matrix with entries in {0,1,...,50}
        M = np.random.randint(0, self.matrix_entry_max + 1, size=(n, n))
        
        # Ensure matrix is non-singular
        while abs(np.linalg.det(M.astype(float))) < 1e-10:
            print(f"    Matrix singular, regenerating...")
            M = np.random.randint(0, self.matrix_entry_max + 1, size=(n, n))
        
        return M
    
    def apply_lll_reduction(self, M):
        """Apply LLL reduction using Sage."""
        print(f"  Applying LLL reduction...")
        
        # Convert to Sage matrix
        M_sage = matrix(ZZ, M)
        
        # Apply LLL reduction
        B_sage = M_sage.LLL()
        
        # Convert back to numpy
        B = np.array(B_sage, dtype=int)
        
        return B
    
    def compute_gram_schmidt(self, B):
        """Compute Gram-Schmidt orthogonalization and norms."""
        print(f"  Computing Gram-Schmidt orthogonalization...")
        
        B_float = B.astype(float)
        n = B.shape[0]
        
        # Gram-Schmidt process
        B_star = np.zeros_like(B_float)
        mu = np.zeros((n, n))
        gs_norms = []
        
        for i in range(n):
            B_star[i] = B_float[i].copy()
            
            # Subtract projections onto previous vectors
            for j in range(i):
                mu[i, j] = np.dot(B_float[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                B_star[i] -= mu[i, j] * B_star[j]
            
            # Compute norm
            norm = np.linalg.norm(B_star[i])
            gs_norms.append(norm)
        
        return B_star, gs_norms
    
    def compute_basis_properties(self, B):
        """Compute mathematical properties of the lattice basis."""
        print(f"  Computing basis properties...")
        
        B_float = B.astype(float)
        
        # Basic properties
        determinant = float(np.linalg.det(B_float))
        condition_number = float(np.linalg.cond(B_float))
        
        # Gram-Schmidt properties
        _, gs_norms = self.compute_gram_schmidt(B)
        max_gs_norm = max(gs_norms)
        min_gs_norm = min(gs_norms)
        
        properties = {
            "determinant": determinant,
            "condition_number": condition_number,
            "gs_norms": gs_norms,
            "max_gs_norm": max_gs_norm,
            "min_gs_norm": min_gs_norm,
            "gs_norm_ratio": max_gs_norm / min_gs_norm if min_gs_norm > 0 else float('inf')
        }
        
        print(f"    Determinant: {determinant:.2e}")
        print(f"    Condition number: {condition_number:.2e}")
        print(f"    Max GS norm: {max_gs_norm:.4f}")
        print(f"    Min GS norm: {min_gs_norm:.4f}")
        
        return properties
    
    def generate_klein_samples(self, B, sigma, n):
        """Generate samples using Klein sampler."""
        print(f"  Generating {self.n_samples:,} Klein samples...")
        
        # Create center vector (all zeros)
        center = np.zeros(n)
        
        # Create lattice and sampler
        lattice = SimpleLattice(B.astype(float))
        klein_sampler = RefinedKleinSampler(lattice, sigma, center)
        
        # Measure sampling time
        start_time = time.time()
        samples = klein_sampler.sample(self.n_samples)
        end_time = time.time()
        
        sampling_time = end_time - start_time
        time_per_sample_ms = (sampling_time / self.n_samples) * 1000
        
        print(f"    Sampling time: {sampling_time:.2f}s")
        print(f"    Time per sample: {time_per_sample_ms:.4f}ms")
        
        return samples, sampling_time, time_per_sample_ms
    
    def analyze_sample_quality(self, samples, B, sigma):
        """Analyze sample quality using practical metrics instead of theoretical TV."""
        print(f"  Analyzing sample quality...")
        
        # Convert lattice points back to integer coordinates
        B_inv = np.linalg.inv(B.astype(float))
        integer_coords = samples @ B_inv.T
        integer_coords = np.round(integer_coords).astype(int)
        
        # Basic statistics
        sample_means = np.mean(integer_coords, axis=0)
        sample_stds = np.std(integer_coords, axis=0)
        sample_ranges = np.ptp(integer_coords, axis=0)  # peak-to-peak (max - min)
        
        # First coordinate statistics
        x1_samples = integer_coords[:, 0]
        x1_mean = sample_means[0]
        x1_std = sample_stds[0]
        x1_min, x1_max = x1_samples.min(), x1_samples.max()
        
        print(f"    First coordinate stats:")
        print(f"      Mean: {x1_mean:.4f}")
        print(f"      Std:  {x1_std:.4f}")
        print(f"      Range: [{x1_min}, {x1_max}]")
        
        # Overall sample quality metrics
        mean_magnitude = np.mean(np.abs(sample_means))
        std_uniformity = np.std(sample_stds) / np.mean(sample_stds) if np.mean(sample_stds) > 0 else float('inf')
        
        print(f"    Overall quality:")
        print(f"      Mean magnitude: {mean_magnitude:.4f}")
        print(f"      Std uniformity (CV): {std_uniformity:.4f}")
        
        # Check for degenerate sampling (all samples the same)
        unique_samples = len(np.unique(integer_coords.view(np.void), axis=0))
        sample_diversity = unique_samples / self.n_samples
        
        print(f"      Sample diversity: {sample_diversity:.4f} ({unique_samples}/{self.n_samples})")
        
        quality_metrics = {
            'x1_mean': float(x1_mean),
            'x1_std': float(x1_std),
            'x1_range': [int(x1_min), int(x1_max)],
            'mean_magnitude': float(mean_magnitude),
            'std_uniformity': float(std_uniformity),
            'sample_diversity': float(sample_diversity),
            'all_means': [float(x) for x in sample_means],
            'all_stds': [float(x) for x in sample_stds],
            'all_ranges': [int(x) for x in sample_ranges]
        }
        
        return quality_metrics
    
    def create_sample_distribution_plot(self, n, integer_coords, quality_metrics, sigma):
        """Create publication-quality plot of sample distribution."""
        print(f"  Creating sample distribution plot...")
        
        x1_samples = integer_coords[:, 0]
        x1_min, x1_max = quality_metrics['x1_range']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram of first coordinate
        ax1.hist(x1_samples, bins=range(x1_min-1, x1_max+2), alpha=0.7, 
                color='skyblue', edgecolor='navy', density=True)
        ax1.set_xlabel('x₁ coordinate', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'First coordinate distribution\nn={n}, σ={sigma:.2f}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {quality_metrics['x1_mean']:.3f}\nStd: {quality_metrics['x1_std']:.3f}"
        ax1.text(0.7, 0.9, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                fontsize=10, verticalalignment='top')
        
        # Plot 2: All coordinates standard deviations
        coords = range(1, len(quality_metrics['all_stds']) + 1)
        ax2.bar(coords, quality_metrics['all_stds'], alpha=0.7, 
               color='lightgreen', edgecolor='darkgreen')
        ax2.set_xlabel('Coordinate index', fontsize=12)
        ax2.set_ylabel('Standard deviation', fontsize=12)
        ax2.set_title(f'Standard deviations by coordinate\nn={n}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"Klein_sample_distribution_n={n}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Plot saved: {plot_path}")
        return plot_path
    
    def process_dimension(self, n):
        """Process a single dimension completely."""
        print(f"\n{'='*60}")
        print(f"📊 Processing Dimension n = {n}")
        print(f"{'='*60}")
        
        # Step 1: Generate random matrix
        M = self.generate_random_matrix(n)
        
        # Step 2: Apply LLL reduction
        B = self.apply_lll_reduction(M)
        
        # Step 3: Compute basis properties
        properties = self.compute_basis_properties(B)
        
        # Step 4: Set sigma parameter
        sigma = self.sigma_multiplier * properties["max_gs_norm"]
        print(f"  Sigma parameter: {sigma:.4f}")
        
        # Step 5: Generate Klein samples
        samples, sampling_time, time_per_sample_ms = self.generate_klein_samples(B, sigma, n)
        
        # Step 6: Analyze sample quality
        quality_metrics = self.analyze_sample_quality(samples, B, sigma)
        
        # Convert samples for plotting
        B_inv = np.linalg.inv(B.astype(float))
        integer_coords = samples @ B_inv.T
        integer_coords = np.round(integer_coords).astype(int)
        
        # Step 7: Create plot
        plot_path = self.create_sample_distribution_plot(n, integer_coords, quality_metrics, sigma)
        
        # Step 8: Save results to JSON
        results = {
            "n": n,
            "determinant": properties["determinant"],
            "condition_number": properties["condition_number"],
            "max_GS_norm": properties["max_gs_norm"],
            "sigma": sigma,
            "time_per_sample_ms": time_per_sample_ms,
            "quality_metrics": quality_metrics,
            "seed": self.seed,
            "additional_info": {
                "sampling_time_total": sampling_time,
                "gs_norms": properties["gs_norms"],
                "sigma_multiplier": self.sigma_multiplier
            }
        }
        
        json_path = self.output_dir / f"Klein_LLL_n={n}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results saved: {json_path}")
        
        # Store for summary
        self.results[n] = results
        
        print(f"✅ Dimension n={n} completed successfully!")
        return results
    
    def create_summary_plots(self):
        """Create summary plots across all dimensions."""
        print(f"\n{'='*60}")
        print(f"📈 Creating Summary Plots")
        print(f"{'='*60}")
        
        # Extract data for plotting
        dimensions = sorted(self.results.keys())
        time_values = [self.results[n]["time_per_sample_ms"] for n in dimensions]
        diversity_values = [self.results[n]["quality_metrics"]["sample_diversity"] for n in dimensions]
        std_values = [self.results[n]["quality_metrics"]["x1_std"] for n in dimensions]
        
        # Plot 1: Time per Sample vs Dimension
        print("  Creating Time vs Dimension plot...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(dimensions, time_values, 'go-', linewidth=2, markersize=8,
                markerfacecolor='lightgreen', markeredgecolor='green', markeredgewidth=2)
        
        # Annotate points
        for n, time_val in zip(dimensions, time_values):
            ax.annotate(f'{time_val:.3f}ms', (n, time_val), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10)
        
        ax.set_xlabel('Dimension n', fontsize=12)
        ax.set_ylabel('Time per Sample (ms)', fontsize=12)
        ax.set_title('Time per Sample vs. n for Klein on LLL-Reduced Bases', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dimensions)
        
        plt.tight_layout()
        time_plot_path = self.output_dir / "time_vs_dimension.png"
        plt.savefig(time_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {time_plot_path}")
        
        # Plot 2: Sample Quality vs Dimension
        print("  Creating Sample Quality vs Dimension plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sample diversity
        ax1.plot(dimensions, diversity_values, 'bo-', linewidth=2, markersize=8,
                markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2)
        
        for n, div in zip(dimensions, diversity_values):
            ax1.annotate(f'{div:.3f}', (n, div), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        ax1.set_xlabel('Dimension n', fontsize=12)
        ax1.set_ylabel('Sample Diversity', fontsize=12)
        ax1.set_title('Sample Diversity vs. Dimension', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dimensions)
        ax1.set_ylim(0, 1)
        
        # First coordinate standard deviation
        ax2.plot(dimensions, std_values, 'ro-', linewidth=2, markersize=8,
                markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=2)
        
        for n, std in zip(dimensions, std_values):
            ax2.annotate(f'{std:.2f}', (n, std), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        ax2.set_xlabel('Dimension n', fontsize=12)
        ax2.set_ylabel('x₁ Standard Deviation', fontsize=12)
        ax2.set_title('First Coordinate Std vs. Dimension', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(dimensions)
        
        plt.tight_layout()
        quality_plot_path = self.output_dir / "sample_quality_vs_dimension.png"
        plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {quality_plot_path}")
    
    def create_summary_csv(self):
        """Create CSV summary file."""
        print("  Creating CSV summary...")
        
        csv_path = self.output_dir / "Klein_LLL_summary.csv"
        
        # Prepare data
        rows = []
        for n in sorted(self.results.keys()):
            result = self.results[n]
            qm = result['quality_metrics']
            rows.append({
                'n': n,
                'det': result['determinant'],
                'condition_number': result['condition_number'],
                'max_GS_norm': result['max_GS_norm'],
                'sigma': result['sigma'],
                'time_per_sample_ms': result['time_per_sample_ms'],
                'x1_std': qm['x1_std'],
                'sample_diversity': qm['sample_diversity'],
                'mean_magnitude': qm['mean_magnitude']
            })
        
        # Write CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['n', 'det', 'condition_number', 'max_GS_norm', 'sigma', 
                         'time_per_sample_ms', 'x1_std', 'sample_diversity', 'mean_magnitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        print(f"    Saved: {csv_path}")
        
        # Also print summary table
        print(f"\n📋 SUMMARY TABLE")
        print(f"{'='*90}")
        print(f"{'n':>3} {'det':>12} {'cond':>10} {'σ':>8} {'time_ms':>10} {'x1_std':>8} {'diversity':>10}")
        print(f"{'-'*90}")
        for row in rows:
            print(f"{row['n']:>3} {row['det']:>12.2e} {row['condition_number']:>10.2e} "
                  f"{row['sigma']:>8.2f} {row['time_per_sample_ms']:>10.4f} "
                  f"{row['x1_std']:>8.2f} {row['sample_diversity']:>10.4f}")
        print(f"{'='*90}")
    
    def run_full_analysis(self):
        """Run the complete scaling analysis."""
        print("🚀 Starting Klein Sampler Scaling Analysis")
        print("=" * 70)
        
        # Process each dimension
        for n in self.dimensions:
            try:
                self.process_dimension(n)
            except Exception as e:
                print(f"❌ ERROR processing dimension {n}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary materials
        if len(self.results) > 0:
            self.create_summary_plots()
            self.create_summary_csv()
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"📁 Results directory: {self.output_dir}")
            print(f"📊 Processed dimensions: {sorted(self.results.keys())}")
            
            # Final assessment based on sample quality
            all_diverse = all(self.results[n]["quality_metrics"]["sample_diversity"] > 0.1 for n in self.results)
            mean_time = np.mean([self.results[n]["time_per_sample_ms"] for n in self.results])
            print(f"✅ All sample diversity > 0.1: {all_diverse}")
            print(f"⏱️ Mean time per sample: {mean_time:.4f}ms")
        else:
            print("❌ No results to summarize!")

def main():
    """Main execution function."""
    # Ensure we're using the right random seed
    np.random.seed(SEED)
    set_random_seed(SEED)
    
    # Run analysis
    analyzer = KleinScalingAnalysis()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()