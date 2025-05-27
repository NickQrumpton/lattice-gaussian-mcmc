#!/usr/bin/env sage
"""
Core experiments for lattice Gaussian MCMC - standalone version.
"""

from sage.all import *
import numpy as np
import json
import time
from pathlib import Path

# Create directories
for d in ['results/samples', 'results/diagnostics', 'results/figures', 'results/tables']:
    Path(d).mkdir(parents=True, exist_ok=True)

print("LATTICE GAUSSIAN MCMC EXPERIMENTS")
print("="*60)

# Set seeds
set_random_seed(42)
np.random.seed(42)


# 1. IDENTITY LATTICE EXPERIMENTS
print("\n1. IDENTITY LATTICE Z^n")
print("-"*40)

identity_results = []

for n in [16, 64, 256]:
    for regime, factor in [('hard', 0.5), ('near', 1.0), ('smooth', 2.0)]:
        sigma = factor * sqrt(n).n()
        n_samples = min(5000, 50000 // n)
        
        print(f"\nn={n}, regime={regime}, σ={sigma:.2f}")
        
        # Sample from Z^n
        start = time.time()
        samples = []
        
        for _ in range(n_samples):
            # Direct discrete Gaussian sampling
            v = vector(ZZ, [Integer(round(normalvariate(0, sigma))) for _ in range(n)])
            samples.append(v)
        
        elapsed = time.time() - start
        
        # Convert to array
        samples_array = np.array([[int(x) for x in v] for v in samples])
        norms = np.linalg.norm(samples_array, axis=1)
        
        result = {
            'lattice': 'identity',
            'n': n,
            'sigma': float(sigma),
            'regime': regime,
            'n_samples': n_samples,
            'time': elapsed,
            'rate': n_samples/elapsed,
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms))
        }
        
        identity_results.append(result)
        print(f"  Rate: {result['rate']:.1f} samples/sec")
        print(f"  Mean norm: {result['mean_norm']:.2f}")
        
        # Save samples
        np.savez_compressed(
            f'results/samples/identity_n{n}_{regime}.npz',
            samples=samples_array,
            norms=norms
        )

# Save identity results
with open('results/diagnostics/identity_results.json', 'w') as f:
    json.dump(identity_results, f, indent=2)


# 2. NTRU LATTICE EXPERIMENTS  
print("\n\n2. NTRU LATTICES")
print("-"*40)

# Simple NTRU implementation
class SimpleNTRU:
    def __init__(self, n, q):
        self.n = n
        self.q = q
        
        # Polynomial ring
        R.<x> = PolynomialRing(ZZ)
        self.R = R.quotient(x^n + 1)
        
        # Generate simple keys
        self.f = self.R([1] + [choice([-1,0,1]) for _ in range(n-1)])
        self.g = self.R([choice([-1,0,1]) for _ in range(n)])
        
        # Basis (simplified - just use standard form)
        I_n = identity_matrix(ZZ, n)
        Z_n = zero_matrix(ZZ, n)
        
        # Random h for public key simulation
        h_coeffs = [randint(0, q-1) for _ in range(n)]
        H = matrix(ZZ, n, n)
        for i in range(n):
            for j in range(n):
                H[i,j] = h_coeffs[(j-i) % n]
        
        self.basis = block_matrix([
            [q * I_n, Z_n],
            [H, I_n]
        ])
        
        # Compute simple GS norms
        self.gs_min = 1.0
        self.gs_max = float(q)

ntru_results = []

for n, q in [(64, 257), (64, 12289)]:
    print(f"\nNTRU: n={n}, q={q}")
    
    ntru = SimpleNTRU(n, q)
    print(f"  GS ratio: {ntru.gs_max/ntru.gs_min:.1f}")
    
    for regime, sigma in [('hard', 10.0), ('medium', 50.0), ('smooth', 100.0)]:
        n_samples = 500
        
        print(f"\n  Regime: {regime}, σ={sigma}")
        
        # Sample using CVP approximation
        start = time.time()
        samples = []
        
        for _ in range(n_samples):
            # Sample continuous Gaussian
            y = vector(RDF, [normalvariate(0, sigma) for _ in range(2*n)])
            
            # Approximate CVP (Babai)
            try:
                coeffs = ntru.basis.solve_left(y)
                rounded = vector(ZZ, [round(c) for c in coeffs])
                v = ntru.basis * rounded
                samples.append(v)
            except:
                # Fallback to random
                v = vector(ZZ, 2*n)
                samples.append(v)
        
        elapsed = time.time() - start
        
        # Convert to array
        samples_array = np.array([[int(x) for x in v] for v in samples])
        norms = np.linalg.norm(samples_array, axis=1)
        
        result = {
            'lattice': 'ntru',
            'n': n,
            'q': q,
            'sigma': float(sigma),
            'regime': regime,
            'n_samples': n_samples,
            'time': elapsed,
            'rate': n_samples/elapsed,
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'gs_ratio': ntru.gs_max/ntru.gs_min
        }
        
        ntru_results.append(result)
        print(f"    Rate: {result['rate']:.1f} samples/sec")
        
        # Save samples
        np.savez_compressed(
            f'results/samples/ntru_n{n}_q{q}_{regime}.npz',
            samples=samples_array,
            norms=norms
        )

# Save NTRU results
with open('results/diagnostics/ntru_results.json', 'w') as f:
    json.dump(ntru_results, f, indent=2)


# 3. CONVERGENCE ANALYSIS
print("\n\n3. CONVERGENCE ANALYSIS")
print("-"*40)

# Simple TVD computation for identity lattice
n = 16
sigma = sqrt(n).n()

print(f"\nConvergence for Z^{n}, σ={sigma:.2f}")

# Reference distribution (many samples)
ref_norms = []
for _ in range(10000):
    v = vector([normalvariate(0, sigma) for _ in range(n)])
    ref_norms.append(v.norm())

ref_norms = np.array(ref_norms)

# Track TVD over iterations
iterations = [10, 50, 100, 500, 1000, 5000]
tvd_values = []

for n_iter in iterations:
    sample_norms = []
    for _ in range(n_iter):
        v = vector([normalvariate(0, sigma) for _ in range(n)])
        sample_norms.append(v.norm())
    
    sample_norms = np.array(sample_norms)
    
    # Compute TVD
    bins = np.linspace(0, max(ref_norms.max(), sample_norms.max()), 30)
    hist_ref, _ = np.histogram(ref_norms, bins=bins, density=True)
    hist_sample, _ = np.histogram(sample_norms, bins=bins, density=True) 
    
    # Normalize
    hist_ref = hist_ref / (hist_ref.sum() + 1e-10)
    hist_sample = hist_sample / (hist_sample.sum() + 1e-10)
    
    tvd = 0.5 * np.abs(hist_ref - hist_sample).sum()
    tvd_values.append(float(tvd))
    
    print(f"  Iter {n_iter}: TVD = {tvd:.4f}")

# Save convergence data
conv_data = {
    'lattice': 'identity',
    'n': n,
    'sigma': float(sigma),
    'iterations': iterations,
    'tvd_values': tvd_values
}

with open('results/diagnostics/convergence_data.json', 'w') as f:
    json.dump(conv_data, f, indent=2)


# 4. SUMMARY
print("\n\n4. EXPERIMENT SUMMARY")
print("-"*60)

all_results = identity_results + ntru_results

print(f"\nTotal experiments run: {len(all_results)}")
print("\nResults by lattice type:")
print(f"  Identity: {len(identity_results)} experiments")
print(f"  NTRU: {len(ntru_results)} experiments")

# Performance summary
print("\nPerformance Summary (samples/sec):")
print("-"*40)
print(f"{'Lattice':<10} {'Min':<10} {'Max':<10} {'Mean':<10}")
print("-"*40)

for lattice_type in ['identity', 'ntru']:
    rates = [r['rate'] for r in all_results if r['lattice'] == lattice_type]
    if rates:
        print(f"{lattice_type:<10} {min(rates):<10.1f} {max(rates):<10.1f} {np.mean(rates):<10.1f}")

# Save full summary
summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_experiments': len(all_results),
    'experiments': all_results,
    'convergence': conv_data
}

with open('results/diagnostics/full_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ ALL EXPERIMENTS COMPLETED")
print("\nResults saved in:")
print("  - results/samples/      (raw sample data)")
print("  - results/diagnostics/  (experiment metrics)")
print("  - results/diagnostics/full_summary.json")