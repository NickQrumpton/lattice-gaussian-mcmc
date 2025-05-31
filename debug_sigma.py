#!/usr/bin/env sage -python
"""
Debug the actual relationship between Klein sampler sigma and output distribution.
"""

import numpy as np
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

def debug_klein_sigma():
    """Analyze the relationship between Klein sigma and output distribution."""
    
    n = 16
    sigma_multiplier = 5.0
    n_samples = 10000
    
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
    sigma = sigma_multiplier * max_gs_norm
    
    print(f"Lattice basis B shape: {B.shape}")
    print(f"Max GS norm: {max_gs_norm:.4f}")
    print(f"Klein sampler sigma: {sigma:.4f}")
    print(f"GS norms: {[f'{x:.2f}' for x in gs_norms[:5]]}...")
    
    # Generate Klein samples
    center = np.zeros(n)
    lattice = SimpleLattice(B.astype(float))
    klein_sampler = RefinedKleinSampler(lattice, sigma, center)
    
    print(f"Generating {n_samples} samples...")
    samples = klein_sampler.sample(n_samples)
    
    print(f"Samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    # Convert to integer coordinates
    B_inv = np.linalg.inv(B.astype(float))
    integer_coords = samples @ B_inv.T
    integer_coords = np.round(integer_coords).astype(int)
    x1_samples = integer_coords[:, 0]
    
    print(f"\\nInteger coordinates analysis:")
    print(f"x1 range: [{x1_samples.min()}, {x1_samples.max()}]")
    print(f"x1 empirical mean: {np.mean(x1_samples):.4f}")
    print(f"x1 empirical std: {np.std(x1_samples):.4f}")
    
    # Now let's try different sigma values and see what standard deviation we get
    print(f"\\n=== Testing different sigma values ===")
    
    test_sigmas = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    for test_sigma in test_sigmas:
        test_lattice = SimpleLattice(B.astype(float))
        test_sampler = RefinedKleinSampler(test_lattice, test_sigma, center)
        test_samples = test_sampler.sample(1000)
        test_coords = test_samples @ B_inv.T
        test_coords = np.round(test_coords).astype(int)
        test_x1 = test_coords[:, 0]
        
        print(f"sigma={test_sigma:5.1f} → x1_std={np.std(test_x1):6.2f}, range=[{test_x1.min():3d}, {test_x1.max():3d}]")
    
    # Try to find the sigma that gives us the empirical std
    target_std = np.std(x1_samples)
    print(f"\\nTarget std (from original): {target_std:.4f}")
    
    # Binary search for the right sigma
    low_sigma, high_sigma = 0.1, 10.0
    for _ in range(10):
        mid_sigma = (low_sigma + high_sigma) / 2
        test_lattice = SimpleLattice(B.astype(float))
        test_sampler = RefinedKleinSampler(test_lattice, mid_sigma, center)
        test_samples = test_sampler.sample(1000)
        test_coords = test_samples @ B_inv.T
        test_coords = np.round(test_coords).astype(int)
        test_x1 = test_coords[:, 0]
        test_std = np.std(test_x1)
        
        print(f"Testing sigma={mid_sigma:.4f} → std={test_std:.4f} (target: {target_std:.4f})")
        
        if test_std < target_std:
            low_sigma = mid_sigma
        else:
            high_sigma = mid_sigma
    
    best_sigma = (low_sigma + high_sigma) / 2
    print(f"\\nEstimated best sigma for theoretical calculation: {best_sigma:.4f}")
    print(f"Original Klein sigma was: {sigma:.4f}")
    print(f"Ratio: {sigma / best_sigma:.2f}")

if __name__ == "__main__":
    debug_klein_sigma()