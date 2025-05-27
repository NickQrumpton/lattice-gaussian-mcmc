#!/usr/bin/env sage
"""
Simplified experiment runner for lattice Gaussian MCMC.

This script runs core experiments on identity and NTRU lattices,
collecting sampling data and diagnostics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sage.all import *
import numpy as np
import json
import time
from pathlib import Path

# Load implementations
load('../src/lattices/ntru_clean.py')
from src.core.discrete_gaussian import sample_discrete_gaussian_vec
from src.lattices.gaussian_lattice_sampler import IdentityLatticeSampler
from src.diagnostics.mcmc_diag import effective_sample_size, compute_autocorrelation


def create_directories():
    """Create output directories."""
    dirs = [
        'results/samples',
        'results/diagnostics', 
        'results/figures',
        'results/tables',
        'results/logs'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def run_identity_lattice_experiments():
    """Run experiments on identity lattice Z^n."""
    print("\n" + "="*60)
    print("IDENTITY LATTICE EXPERIMENTS")
    print("="*60)
    
    results = []
    
    # Test dimensions
    dimensions = [16, 64, 256]
    
    # Sigma regimes (relative to sqrt(n))
    sigma_factors = {
        'hard': 0.5,
        'near': 1.0,
        'smooth': 2.0
    }
    
    for n in dimensions:
        print(f"\n[Identity] Dimension n = {n}")
        
        for regime, factor in sigma_factors.items():
            sigma = factor * sqrt(n)
            n_samples = min(10000, 100000 // n)  # Adjust for dimension
            
            print(f"  Regime: {regime}, σ = {sigma:.2f}, samples = {n_samples}")
            
            # Create sampler
            sampler = IdentityLatticeSampler(n=n, sigma=float(sigma))
            
            # Sample with timing
            start_time = time.time()
            samples = []
            
            for i in range(n_samples):
                v = sampler.sample()
                samples.append(v)
                
                if i > 0 and i % 1000 == 0:
                    print(f"    Progress: {i}/{n_samples}")
            
            elapsed = time.time() - start_time
            
            # Convert to numpy
            samples_array = np.array([[float(x) for x in v] for v in samples])
            
            # Compute diagnostics
            norms = np.linalg.norm(samples_array, axis=1)
            first_coord = samples_array[:, 0]
            
            # ESS and autocorrelation
            ess = effective_sample_size(first_coord)
            acf = compute_autocorrelation(first_coord, max_lag=50)
            
            # Results
            result = {
                'lattice_type': 'identity',
                'dimension': n,
                'sigma': float(sigma),
                'sigma_regime': regime,
                'n_samples': n_samples,
                'elapsed_time': elapsed,
                'samples_per_sec': n_samples / elapsed,
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'ess': float(ess),
                'ess_per_sample': float(ess / n_samples),
                'acf_lag_1': float(acf[1]) if len(acf) > 1 else 0.0,
                'acf_lag_10': float(acf[10]) if len(acf) > 10 else 0.0
            }
            
            results.append(result)
            
            # Save samples
            filename = f"results/samples/identity_n{n}_{regime}.npz"
            np.savez_compressed(filename, 
                               samples=samples_array,
                               norms=norms,
                               metadata=result)
            
            print(f"    ESS/n = {result['ess_per_sample']:.3f}")
            print(f"    Rate = {result['samples_per_sec']:.1f} samples/sec")
    
    # Save all results
    with open('results/diagnostics/identity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_ntru_lattice_experiments():
    """Run experiments on NTRU lattices."""
    print("\n" + "="*60)
    print("NTRU LATTICE EXPERIMENTS")
    print("="*60)
    
    results = []
    
    # NTRU parameters
    ntru_configs = [
        {'n': 64, 'q': 257},
        {'n': 64, 'q': 12289},
        # {'n': 512, 'q': 12289}  # Uncomment for full run
    ]
    
    sigma_factors = {
        'hard': 10.0,
        'medium': 50.0,
        'smooth': 100.0
    }
    
    for config in ntru_configs:
        n, q = config['n'], config['q']
        print(f"\n[NTRU] n = {n}, q = {q}")
        
        # Create NTRU lattice
        ntru = NTRULattice(n=n, q=q)
        print("  Generating keys...")
        
        if not ntru.generate_keys(key_type='ternary'):
            print("  ✗ Key generation failed, skipping")
            continue
        
        print("  ✓ Keys generated")
        
        # Get Gram-Schmidt norms
        gs_norms = ntru.gram_schmidt_norms()
        gs_ratio = max(gs_norms) / min(gs_norms)
        print(f"  GS ratio: {gs_ratio:.2f}")
        
        for regime, sigma in sigma_factors.items():
            n_samples = 1000  # Smaller for NTRU due to cost
            
            print(f"\n  Regime: {regime}, σ = {sigma}, samples = {n_samples}")
            
            # Sample with timing
            start_time = time.time()
            samples = []
            
            for i in range(n_samples):
                v = ntru.sample_discrete_gaussian(sigma=sigma)
                samples.append(v)
                
                if i > 0 and i % 100 == 0:
                    print(f"    Progress: {i}/{n_samples}")
            
            elapsed = time.time() - start_time
            
            # Convert to numpy
            samples_array = np.array([[float(x) for x in v] for v in samples])
            
            # Compute diagnostics
            norms = np.linalg.norm(samples_array, axis=1)
            
            # Check lattice membership for first few
            n_check = min(10, n_samples)
            B = ntru.get_basis()
            in_lattice = 0
            
            for i in range(n_check):
                try:
                    coeffs = B.solve_left(vector(samples[i]))
                    if all(abs(c - round(c)) < 1e-10 for c in coeffs):
                        in_lattice += 1
                except:
                    pass
            
            # Results
            result = {
                'lattice_type': 'ntru',
                'dimension': n,
                'q': q,
                'sigma': float(sigma),
                'sigma_regime': regime,
                'n_samples': n_samples,
                'elapsed_time': elapsed,
                'samples_per_sec': n_samples / elapsed,
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'gs_ratio': float(gs_ratio),
                'lattice_membership_rate': in_lattice / n_check
            }
            
            results.append(result)
            
            # Save samples
            filename = f"results/samples/ntru_n{n}_q{q}_{regime}.npz"
            np.savez_compressed(filename,
                               samples=samples_array,
                               norms=norms,
                               metadata=result)
            
            print(f"    Rate = {result['samples_per_sec']:.1f} samples/sec")
            print(f"    Lattice membership = {result['lattice_membership_rate']*100:.0f}%")
    
    # Save all results
    with open('results/diagnostics/ntru_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_convergence_experiments():
    """Run convergence experiments to measure TVD over iterations."""
    print("\n" + "="*60)
    print("CONVERGENCE EXPERIMENTS")
    print("="*60)
    
    results = []
    
    # Simple test: Identity lattice convergence
    n = 16
    sigma = 2.0 * sqrt(n)
    
    print(f"\n[Convergence] Identity lattice n={n}, σ={sigma:.2f}")
    
    # Generate reference samples (ground truth)
    print("  Generating reference samples...")
    sampler = IdentityLatticeSampler(n=n, sigma=float(sigma))
    
    ref_samples = []
    for _ in range(10000):
        ref_samples.append(sampler.sample())
    
    ref_array = np.array([[float(x) for x in v] for v in ref_samples])
    ref_norms = np.linalg.norm(ref_array, axis=1)
    
    # Track convergence over iterations
    iterations = [10, 50, 100, 500, 1000, 5000]
    tvd_values = []
    
    for n_iter in iterations:
        # Generate samples
        samples = []
        for _ in range(n_iter):
            samples.append(sampler.sample())
        
        samples_array = np.array([[float(x) for x in v] for v in samples])
        sample_norms = np.linalg.norm(samples_array, axis=1)
        
        # Compute TVD using histogram approximation
        bins = np.linspace(0, max(ref_norms.max(), sample_norms.max()), 30)
        
        hist_ref, _ = np.histogram(ref_norms, bins=bins, density=True)
        hist_sample, _ = np.histogram(sample_norms, bins=bins, density=True)
        
        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_sample = hist_sample / hist_sample.sum()
        
        tvd = 0.5 * np.abs(hist_ref - hist_sample).sum()
        tvd_values.append(float(tvd))
        
        print(f"  Iteration {n_iter}: TVD = {tvd:.4f}")
    
    # Save convergence data
    convergence_data = {
        'lattice_type': 'identity',
        'dimension': n,
        'sigma': float(sigma),
        'iterations': iterations,
        'tvd_values': tvd_values
    }
    
    with open('results/diagnostics/convergence_data.json', 'w') as f:
        json.dump(convergence_data, f, indent=2)
    
    return convergence_data


def generate_summary_table():
    """Generate summary table of all experiments."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY")
    print("="*60)
    
    # Load results
    all_results = []
    
    if os.path.exists('results/diagnostics/identity_results.json'):
        with open('results/diagnostics/identity_results.json', 'r') as f:
            all_results.extend(json.load(f))
    
    if os.path.exists('results/diagnostics/ntru_results.json'):
        with open('results/diagnostics/ntru_results.json', 'r') as f:
            all_results.extend(json.load(f))
    
    # Create summary table
    print("\nSummary Table:")
    print("-" * 80)
    print(f"{'Lattice':<10} {'Dim':<6} {'Sigma':<10} {'Regime':<10} {'Rate':<15} {'ESS/n':<10}")
    print("-" * 80)
    
    for r in all_results:
        lattice = r['lattice_type']
        dim = r['dimension']
        sigma = r['sigma']
        regime = r['sigma_regime']
        rate = r['samples_per_sec']
        ess_ratio = r.get('ess_per_sample', 'N/A')
        
        if isinstance(ess_ratio, float):
            ess_str = f"{ess_ratio:.3f}"
        else:
            ess_str = ess_ratio
        
        print(f"{lattice:<10} {dim:<6} {sigma:<10.2f} {regime:<10} {rate:<15.1f} {ess_str:<10}")
    
    # Save summary
    summary = {
        'total_experiments': len(all_results),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiments': all_results
    }
    
    with open('results/diagnostics/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("-" * 80)
    print(f"Total experiments: {len(all_results)}")


def main():
    """Run all experiments."""
    print("\n" + "="*60)
    print("LATTICE GAUSSIAN MCMC EXPERIMENTS")
    print("="*60)
    
    # Set random seed
    set_random_seed(42)
    np.random.seed(42)
    
    # Create directories
    create_directories()
    
    # Run experiments
    try:
        # Identity lattice
        identity_results = run_identity_lattice_experiments()
        
        # NTRU lattice
        ntru_results = run_ntru_lattice_experiments()
        
        # Convergence study
        convergence_data = run_convergence_experiments()
        
        # Generate summary
        generate_summary_table()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ EXPERIMENTS COMPLETED")
    print("="*60)
    print("\nResults saved in:")
    print("  - results/samples/      (raw data)")
    print("  - results/diagnostics/  (metrics)")


if __name__ == "__main__":
    main()