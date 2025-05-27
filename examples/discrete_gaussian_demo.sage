#!/usr/bin/env sage
"""
Demonstration of discrete Gaussian sampling on various lattices.

This script shows how to use the exact discrete Gaussian samplers
with identity, q-ary, and NTRU lattices.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sage.all import *
from core.discrete_gaussian import (
    RejectionSampler, CDTSampler, 
    sample_discrete_gaussian_1d, sample_discrete_gaussian_vec
)
from lattices.gaussian_lattice_sampler import (
    IdentityLatticeSampler, QaryLatticeSampler, NTRULatticeSampler
)

# Load NTRU implementation
load('../src/lattices/ntru_clean.py')

import matplotlib.pyplot as plt
import numpy as np


def demo_1d_samplers():
    """Demonstrate 1D discrete Gaussian samplers."""
    print("=" * 60)
    print("1D Discrete Gaussian Sampling")
    print("=" * 60)
    
    # Compare rejection and CDT samplers
    sigma = 2.5
    center = 1.3
    n_samples = 10000
    
    print(f"\nParameters: σ = {sigma}, c = {center}")
    
    # Rejection sampler
    print("\nRejection Sampler:")
    rej_sampler = RejectionSampler(sigma=sigma, center=center)
    rej_samples = rej_sampler.sample(n_samples)
    
    rej_mean = float(mean(rej_samples))
    rej_std = float(std(rej_samples))
    print(f"  Mean: {rej_mean:.4f} (expected: {center})")
    print(f"  Std dev: {rej_std:.4f} (expected: ~{sigma})")
    
    # CDT sampler
    print("\nCDT Sampler:")
    cdt_sampler = CDTSampler(sigma=sigma, center=center)
    cdt_samples = cdt_sampler.sample(n_samples)
    
    cdt_mean = float(mean(cdt_samples))
    cdt_std = float(std(cdt_samples))
    print(f"  Mean: {cdt_mean:.4f}")
    print(f"  Std dev: {cdt_std:.4f}")
    
    # Plot histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(rej_samples, bins=50, density=True, alpha=0.7, label='Rejection')
    plt.hist(cdt_samples, bins=50, density=True, alpha=0.7, label='CDT')
    
    # Overlay theoretical curve
    x_range = range(int(center - 4*sigma), int(center + 4*sigma) + 1)
    probs = [rej_sampler.probability(x) for x in x_range]
    plt.plot(x_range, probs, 'k-', linewidth=2, label='Theoretical')
    
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('1D Discrete Gaussian Samplers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot log probabilities
    plt.subplot(1, 2, 2)
    log_probs = [rej_sampler.log_probability(x) for x in x_range]
    plt.plot(x_range, log_probs, 'b-', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Log Probability')
    plt.title('Log Probability Function')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/discrete_gaussian_1d.png', dpi=150)
    print("\n✓ Saved plot to results/figures/discrete_gaussian_1d.png")


def demo_identity_lattice():
    """Demonstrate sampling on identity lattice Z^n."""
    print("\n" + "=" * 60)
    print("Identity Lattice Z^n Sampling")
    print("=" * 60)
    
    n = 10
    sigma = 2.0
    
    print(f"\nParameters: n = {n}, σ = {sigma}")
    
    # Create sampler
    sampler = IdentityLatticeSampler(n=n, sigma=sigma)
    
    # Generate samples
    n_samples = 1000
    samples = sampler.sample(n_samples)
    
    # Compute statistics
    norms = [float(v.norm()) for v in samples]
    mean_norm = mean(norms)
    
    # Theoretical expected norm: ~σ√n
    expected_norm = sigma * sqrt(n)
    
    print(f"\nStatistics over {n_samples} samples:")
    print(f"  Mean norm: {mean_norm:.4f}")
    print(f"  Expected norm: {expected_norm:.4f}")
    print(f"  Min norm: {min(norms):.4f}")
    print(f"  Max norm: {max(norms):.4f}")
    
    # Test with non-uniform sigmas
    sigmas = [1.0 + 0.5*i for i in range(n)]
    nonuniform_sampler = IdentityLatticeSampler(n=n, sigma=sigmas)
    v = nonuniform_sampler.sample()
    print(f"\nNon-uniform σ sample: {v}")
    
    # Plot norm distribution
    plt.figure(figsize=(8, 6))
    plt.hist(norms, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(expected_norm, color='red', linestyle='--', 
                linewidth=2, label=f'Expected: {expected_norm:.2f}')
    plt.xlabel('Vector Norm')
    plt.ylabel('Density')
    plt.title(f'Identity Lattice Z^{n} Sample Norms (σ={sigma})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/figures/identity_lattice_norms.png', dpi=150)
    print("✓ Saved plot to results/figures/identity_lattice_norms.png")


def demo_ntru_lattice():
    """Demonstrate sampling on NTRU lattice."""
    print("\n" + "=" * 60)
    print("NTRU Lattice Sampling")
    print("=" * 60)
    
    # Create NTRU lattice
    n = 64
    q = 12289
    print(f"\nCreating NTRU lattice: n = {n}, q = {q}")
    
    ntru = NTRULattice(n=n, q=q)
    print("Generating keys...")
    if not ntru.generate_keys(key_type='ternary'):
        print("✗ Key generation failed")
        return
    
    print("✓ Keys generated")
    
    # Create Gaussian sampler
    sigma = 100.0  # Typical for NTRU
    sampler = NTRULatticeSampler(ntru, sigma=sigma)
    
    # Generate samples
    print(f"\nSampling with σ = {sigma}")
    n_samples = 100
    samples = []
    
    for i in range(n_samples):
        v = sampler.sample()
        samples.append(v)
        if i == 0:
            print(f"First sample norm: {float(v.norm()):.2f}")
    
    # Analyze samples
    norms = [float(v.norm()) for v in samples]
    
    print(f"\nStatistics over {n_samples} samples:")
    print(f"  Mean norm: {mean(norms):.2f}")
    print(f"  Std dev: {std(norms):.2f}")
    print(f"  Min norm: {min(norms):.2f}")
    print(f"  Max norm: {max(norms):.2f}")
    
    # Verify samples are in lattice
    B = ntru.get_basis()
    for i in range(min(5, n_samples)):
        try:
            coeffs = B.solve_left(samples[i])
            is_integral = all(abs(c - round(c)) < 1e-10 for c in coeffs)
            print(f"  Sample {i} in lattice: {is_integral}")
        except:
            print(f"  Sample {i} verification failed")
    
    # Compare with short vector sampling
    print("\nShort vector sampling (σ/2):")
    short_samples = []
    for _ in range(10):
        v = sampler.sample_short_vector()
        short_samples.append(v)
    
    short_norms = [float(v.norm()) for v in short_samples]
    print(f"  Mean short norm: {mean(short_norms):.2f}")
    
    # Plot both distributions
    plt.figure(figsize=(10, 6))
    
    plt.hist(norms, bins=20, density=True, alpha=0.7, 
             label=f'Standard (σ={sigma})', color='blue')
    plt.hist(short_norms, bins=10, density=True, alpha=0.7,
             label=f'Short (σ={sigma/2})', color='red')
    
    plt.xlabel('Vector Norm')
    plt.ylabel('Density')
    plt.title(f'NTRU Lattice Sample Norms (n={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/figures/ntru_lattice_norms.png', dpi=150)
    print("✓ Saved plot to results/figures/ntru_lattice_norms.png")


def demo_performance():
    """Benchmark performance of different samplers."""
    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)
    
    import time
    
    # Test different sigma values
    sigmas = [1.0, 5.0, 10.0, 50.0, 100.0]
    n_samples = 10000
    
    print(f"\nSampling {n_samples} values for each σ:")
    print("-" * 40)
    print("σ      Rejection(ms)  CDT(ms)  Ratio")
    print("-" * 40)
    
    for sigma in sigmas:
        # Rejection sampler
        rej_sampler = RejectionSampler(sigma=sigma)
        start = time.time()
        rej_sampler.sample(n_samples)
        rej_time = (time.time() - start) * 1000
        
        # CDT sampler
        cdt_sampler = CDTSampler(sigma=sigma)
        start = time.time()
        cdt_sampler.sample(n_samples)
        cdt_time = (time.time() - start) * 1000
        
        ratio = rej_time / cdt_time if cdt_time > 0 else float('inf')
        
        print(f"{sigma:<6.1f} {rej_time:>12.1f}  {cdt_time:>8.1f}  {ratio:>5.2f}x")
    
    # Test vector sampling
    print("\n\nVector sampling performance:")
    print("-" * 40)
    print("n      Time(ms)  Rate(vec/s)")
    print("-" * 40)
    
    dimensions = [10, 50, 100, 500, 1000]
    sigma = 10.0
    
    for n in dimensions:
        sampler = IdentityLatticeSampler(n=n, sigma=sigma)
        
        start = time.time()
        samples = sampler.sample(100)
        elapsed = (time.time() - start) * 1000
        rate = 100 / (elapsed / 1000)
        
        print(f"{n:<6} {elapsed:>8.1f}  {rate:>10.1f}")


def main():
    """Run all demonstrations."""
    print("\nDiscrete Gaussian Sampling Demonstrations")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('../results/figures', exist_ok=True)
    
    # Run demos
    demo_1d_samplers()
    demo_identity_lattice()
    demo_ntru_lattice()
    demo_performance()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations completed!")
    print("Check results/figures/ for generated plots")


if __name__ == "__main__":
    main()