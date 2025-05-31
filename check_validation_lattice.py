#!/usr/bin/env sage -python
"""
Check the properties of the validation lattice that worked.
"""

import numpy as np
import sys
sys.path.append('.')

# SageMath imports
from sage.all import *

# Local imports
from src.lattices.simple import SimpleLattice
from src.samplers.klein import RefinedKleinSampler

def check_validation_lattice():
    """Check the 2D validation lattice properties."""
    
    # The successful 2D validation lattice
    basis = np.array([[4.0, 1.0], [1.0, 3.0]])
    sigma = 2.0
    
    print("=== 2D Validation Lattice ===")
    print(f"Basis:\n{basis}")
    print(f"Sigma: {sigma}")
    
    # Compute Gram-Schmidt norms
    B_float = basis.astype(float)
    n = B_float.shape[0]
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
    print(f"GS norms: {gs_norms}")
    print(f"Max GS norm: {max_gs_norm:.4f}")
    
    # Check Klein sampler properties
    lattice = SimpleLattice(basis)
    klein_sampler = RefinedKleinSampler(lattice, sigma, np.zeros(n))
    
    # This will print the smoothing parameter
    print(f"\\nKlein sampler initialized successfully")
    
    # Now compare with the 16D "hard" lattice
    print(f"\\n=== 16D Scaling Analysis Lattice ===")
    
    # Generate the same random matrix as scaling analysis
    SEED = 0
    np.random.seed(SEED + 16)
    M = np.random.randint(0, 51, size=(16, 16))
    while abs(np.linalg.det(M.astype(float))) < 1e-10:
        M = np.random.randint(0, 51, size=(16, 16))
    
    # Apply LLL
    M_sage = matrix(ZZ, M)
    B_sage = M_sage.LLL()
    B = np.array(B_sage, dtype=int)
    
    print(f"Matrix shape: {B.shape}")
    
    # Compute its GS norms
    B_float = B.astype(float)
    n = B_float.shape[0] 
    gs_norms_16d = []
    B_star = np.zeros_like(B_float)
    mu = np.zeros((n, n))
    
    for i in range(n):
        B_star[i] = B_float[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(B_float[i], B_star[j]) / np.dot(B_star[j], B_star[j])
            B_star[i] -= mu[i, j] * B_star[j]
        gs_norms_16d.append(np.linalg.norm(B_star[i]))
    
    max_gs_norm_16d = max(gs_norms_16d)
    print(f"Max GS norm: {max_gs_norm_16d:.4f}")
    print(f"Min GS norm: {min(gs_norms_16d):.4f}")
    print(f"GS norm ratio: {max_gs_norm_16d / min(gs_norms_16d):.2f}")
    
    # Check Klein sampler requirements
    sigma_16d = 5.0 * max_gs_norm_16d
    print(f"Scaling analysis σ: 5.0 × {max_gs_norm_16d:.4f} = {sigma_16d:.4f}")
    
    try:
        lattice_16d = SimpleLattice(B.astype(float))
        klein_sampler_16d = RefinedKleinSampler(lattice_16d, sigma_16d, np.zeros(16))
        print(f"✅ Klein sampler 16D works with σ={sigma_16d:.4f}")
    except Exception as e:
        print(f"❌ Klein sampler 16D failed: {e}")
    
    # Try with smaller sigma on 16D lattice
    test_sigmas = [40.0, 50.0, 60.0, 80.0, 100.0]
    print(f"\\n=== Testing smaller σ values on 16D lattice ===")
    
    for test_sigma in test_sigmas:
        try:
            lattice_16d = SimpleLattice(B.astype(float))
            klein_sampler_16d = RefinedKleinSampler(lattice_16d, test_sigma, np.zeros(16))
            print(f"σ={test_sigma:5.1f}: ✅ Works")
        except Exception as e:
            print(f"σ={test_sigma:5.1f}: ❌ {e}")

if __name__ == "__main__":
    check_validation_lattice()