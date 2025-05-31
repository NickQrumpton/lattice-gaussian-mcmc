#!/usr/bin/env sage -python
"""
Quick test to verify the Klein marginal fix for dimension 16.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import sys
sys.path.append('.')

# SageMath imports
from sage.all import *
from sage.modules.free_module_integer import IntegerLattice

# Local imports
from src.lattices.simple import SimpleLattice
from src.samplers.klein import RefinedKleinSampler

# Set seed for reproducibility
SEED = 0
np.random.seed(SEED)
set_random_seed(SEED)

def test_klein_marginal_fix():
    """Test the fixed marginal computation for n=16."""
    print("Testing Klein marginal fix for n=16...")
    
    n = 16
    sigma_multiplier = 5.0
    n_samples = 10000  # Smaller sample size for quick test
    
    # Generate the same random matrix as in the original (with same seed)
    np.random.seed(SEED + n)
    M = np.random.randint(0, 51, size=(n, n))
    
    # Ensure non-singular
    while abs(np.linalg.det(M.astype(float))) < 1e-10:
        M = np.random.randint(0, 51, size=(n, n))
    
    print(f"Generated {n}x{n} matrix")
    
    # Apply LLL
    M_sage = matrix(ZZ, M)
    B_sage = M_sage.LLL()
    B = np.array(B_sage, dtype=int)
    
    print("Applied LLL reduction")
    
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
    sigma = sigma_multiplier * max_gs_norm
    
    print(f"Max GS norm: {max_gs_norm:.4f}")
    print(f"Sigma: {sigma:.4f}")
    
    # Generate Klein samples
    center = np.zeros(n)
    lattice = SimpleLattice(B.astype(float))
    klein_sampler = RefinedKleinSampler(lattice, sigma, center)
    
    print(f"Generating {n_samples} samples...")
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
    
    print(f"Evaluation interval: [{j_min}, {j_max}] (size: {len(H_n)})")
    
    # Compute empirical probabilities
    empirical_counts = {j: 0 for j in H_n}
    for x1 in x1_samples:
        j = int(round(x1))
        if j in empirical_counts:
            empirical_counts[j] += 1
    
    empirical_probs = {j: count / n_samples for j, count in empirical_counts.items()}
    
    # NEW: Compute theoretical probabilities with FIXED sigma
    sigma_1d = sigma  # Use sigma directly, NOT sigma * gs_norms[0]
    print(f"1D marginal σ (FIXED): {sigma_1d:.4f}")
    
    theoretical_unnorm = {}
    for j in H_n:
        theoretical_unnorm[j] = np.exp(-j**2 / (2 * sigma_1d**2))
    
    Z_1d = sum(theoretical_unnorm.values())
    theoretical_probs = {j: prob / Z_1d for j, prob in theoretical_unnorm.items()}
    
    # Compute TV distance
    tv_distance = 0.5 * sum(abs(empirical_probs[j] - theoretical_probs[j]) for j in H_n)
    
    print(f"\nRESULTS:")
    print(f"TV distance (FIXED): {tv_distance:.6f}")
    print(f"Original TV was: 0.5656 (BAD)")
    print(f"Target TV < 0.02 (GOOD)")
    print(f"Fixed? {'YES' if tv_distance < 0.02 else 'NO'}")
    
    # Create a quick plot
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
    ax.set_title(f'1D marginal (FIXED) n={n}, TV={tv_distance:.4f}, σ={sigma:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/Klein_1D_marginal_FIXED_n=16.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: results/Klein_1D_marginal_FIXED_n=16.png")
    
    return tv_distance

if __name__ == "__main__":
    tv = test_klein_marginal_fix()