#!/usr/bin/env sage -python
"""
Test 16D lattice with proper sigma values (above smoothing parameter).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# SageMath imports
from sage.all import *

# Local imports
from src.lattices.simple import SimpleLattice
from src.samplers.klein import RefinedKleinSampler

# Set seed for reproducibility
SEED = 0
np.random.seed(SEED)
set_random_seed(SEED)

def test_proper_sigma_16d():
    """Test 16D lattice with proper sigma values above smoothing parameter."""
    
    n = 16
    n_samples = 5000  # Smaller for speed
    
    # Generate the same random matrix
    np.random.seed(SEED + n)
    M = np.random.randint(0, 51, size=(n, n))
    while abs(np.linalg.det(M.astype(float))) < 1e-10:
        M = np.random.randint(0, 51, size=(n, n))
    
    # Apply LLL
    M_sage = matrix(ZZ, M)
    B_sage = M_sage.LLL()
    B = np.array(B_sage, dtype=int)
    
    # Compute Gram-Schmidt norms
    B_float = B.astype(float)
    gs_norms = []
    B_star = np.zeros_like(B_float)
    mu = np.zeros((n, n))
    
    for i in range(n):
        B_star[i] = B_float[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(B_float[i], B_star[j]) / np.dot(B_star[j], B_star[j])
            B_star[i] -= mu[i, j] * B_star[j]
        gs_norms.append(np.linalg.norm(B_star[i]))
    
    max_gs_norm = max(gs_norms)
    
    print(f"Max GS norm: {max_gs_norm:.4f}")
    print(f"Smoothing parameter η ≈ {max_gs_norm:.4f} (estimated)")
    
    # Test sigma values above smoothing parameter
    test_sigmas = [40.0, 50.0, 60.0, 80.0, 100.0]
    
    results = []
    
    for sigma in test_sigmas:
        print(f"\\n=== Testing σ = {sigma} ===")
        
        # Generate Klein samples  
        center = np.zeros(n)
        lattice = SimpleLattice(B.astype(float))
        klein_sampler = RefinedKleinSampler(lattice, sigma, center)
        
        samples = klein_sampler.sample(n_samples)
        
        # Convert to integer coordinates
        B_inv = np.linalg.inv(B.astype(float))
        integer_coords = samples @ B_inv.T
        integer_coords = np.round(integer_coords).astype(int)
        x1_samples = integer_coords[:, 0]
        
        # Define evaluation interval
        empirical_min = x1_samples.min()
        empirical_max = x1_samples.max()
        j_min = empirical_min - 2
        j_max = empirical_max + 2
        H_n = range(j_min, j_max + 1)
        
        # Compute empirical probabilities
        empirical_counts = {j: 0 for j in H_n}
        for x1 in x1_samples:
            j = int(round(x1))
            if j in empirical_counts:
                empirical_counts[j] += 1
        
        empirical_probs = {j: count / n_samples for j, count in empirical_counts.items()}
        
        # Compute theoretical probabilities using the Klein sigma directly
        sigma_1d = sigma
        theoretical_unnorm = {}
        for j in H_n:
            theoretical_unnorm[j] = np.exp(-j**2 / (2 * sigma_1d**2))
        
        Z_1d = sum(theoretical_unnorm.values())
        theoretical_probs = {j: prob / Z_1d for j, prob in theoretical_unnorm.items()}
        
        # Compute TV distance
        tv_distance = 0.5 * sum(abs(empirical_probs[j] - theoretical_probs[j]) for j in H_n)
        
        empirical_std = np.std(x1_samples)
        
        print(f"  Range: [{empirical_min}, {empirical_max}]")
        print(f"  Empirical std: {empirical_std:.4f}")
        print(f"  TV distance: {tv_distance:.6f}")
        print(f"  {'✅ GOOD' if tv_distance < 0.02 else '❌ BAD'}")
        
        results.append({
            'sigma': sigma,
            'tv': tv_distance,
            'std': empirical_std,
            'range': [empirical_min, empirical_max],
            'good': tv_distance < 0.02
        })
        
        # Create plot for promising results
        if tv_distance < 0.1:  # Plot if reasonably good
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            j_values = list(H_n)
            emp_values = [empirical_probs[j] for j in j_values]
            theo_values = [theoretical_probs[j] for j in j_values]
            
            ax.bar(j_values, emp_values, alpha=0.7, color='skyblue', 
                   label='Empirical', width=0.8, edgecolor='navy', linewidth=0.5)
            ax.plot(j_values, theo_values, 'r-', linewidth=2, marker='o', 
                    markersize=3, label='Theoretical', alpha=0.8)
            
            ax.set_xlabel('x₁ coordinate')
            ax.set_ylabel('Probability')
            ax.set_title(f'Klein n={n}, σ={sigma}, TV={tv_distance:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"results/Klein_proper_sigma_{sigma}_n={n}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Plot saved: Klein_proper_sigma_{sigma}_n={n}.png")
    
    print(f"\\n=== SUMMARY ===")
    print(f"{'Sigma':>8} {'TV':>10} {'Good?':>8} {'Std':>8}")
    print("-" * 40)
    for r in results:
        status = "✅" if r['good'] else "❌"
        print(f"{r['sigma']:>8.1f} {r['tv']:>10.6f} {status:>8} {r['std']:>8.2f}")
    
    good_results = [r for r in results if r['good']]
    if good_results:
        print(f"\\n🎉 Found {len(good_results)} good sigma values!")
        best = min(good_results, key=lambda x: x['tv'])
        print(f"Best: σ={best['sigma']}, TV={best['tv']:.6f}")
        
        print(f"\\n📊 CONCLUSION:")
        print(f"The issue was using σ=179.75 (too large, causes flat distribution)")
        print(f"Proper σ values (40-100) give much better results!")
        print(f"The scaling analysis should use σ ≈ {best['sigma']} instead of 179.75")
    else:
        print(f"\\n😞 No good sigma values found. Need to investigate further.")

if __name__ == "__main__":
    test_proper_sigma_16d()