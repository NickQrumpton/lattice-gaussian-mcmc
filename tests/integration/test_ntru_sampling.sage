#!/usr/bin/env sage
"""
Integration test for NTRU lattice with Gaussian sampling.

Tests the complete pipeline of key generation, lattice construction,
and Gaussian sampling for cryptographic applications.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sage.all import *
from src.lattices.ntru import NTRULattice
import numpy as np
import matplotlib.pyplot as plt

def test_ntru_gaussian_sampling():
    """Test Gaussian sampling on NTRU lattice."""
    print("Testing NTRU lattice Gaussian sampling...")
    
    # Use small parameters for testing
    n = 64
    q = 257
    sigma = 1.2
    
    print(f"\nParameters: n={n}, q={q}, sigma={sigma}")
    
    # Create NTRU lattice
    lattice = NTRULattice(n=n, q=q, sigma=sigma)
    
    # Generate keys
    print("Generating NTRU keys...")
    keys = lattice.generate_keys(max_attempts=10)
    if keys is None:
        print("❌ Key generation failed")
        return False
    
    f, g, F, G, h = keys
    print("✓ Keys generated successfully")
    
    # Verify lattice properties
    print("\nVerifying lattice properties...")
    B = lattice.basis_matrix
    
    # Check determinant
    det = abs(B.det())
    expected_det = q^n
    print(f"  Determinant: {det} (expected: {expected_det})")
    assert det == expected_det
    
    # Check Gram-Schmidt norms
    gs_norms = lattice.gram_schmidt_norms()
    print(f"  Min GS norm: {min(gs_norms):.4f}")
    print(f"  Max GS norm: {max(gs_norms):.4f}")
    print(f"  GS norm ratio: {max(gs_norms)/min(gs_norms):.4f}")
    
    # Sample from discrete Gaussian
    print("\nSampling from discrete Gaussian...")
    num_samples = 1000
    samples = []
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"  Generated {i}/{num_samples} samples...")
        
        # Sample continuous Gaussian
        c = vector([sigma * normalvariate(0, 1) for _ in range(2*n)])
        
        # Find closest lattice point (using Babai's algorithm)
        v = lattice.closest_vector(c)
        samples.append(v)
    
    print(f"✓ Generated {num_samples} samples")
    
    # Analyze sample statistics
    print("\nAnalyzing sample statistics...")
    
    # Convert to numpy for easier analysis
    samples_np = np.array([[float(x) for x in sample] for sample in samples])
    
    # Compute norms
    norms = np.linalg.norm(samples_np, axis=1)
    print(f"  Average norm: {np.mean(norms):.4f}")
    print(f"  Std dev of norms: {np.std(norms):.4f}")
    print(f"  Min norm: {np.min(norms):.4f}")
    print(f"  Max norm: {np.max(norms):.4f}")
    
    # Check that samples are in the lattice
    print("\nVerifying samples are in lattice...")
    for i in range(min(10, num_samples)):
        assert lattice.is_in_lattice(samples[i]), f"Sample {i} not in lattice!"
    print("✓ All checked samples are in the lattice")
    
    # Plot norm distribution
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Norm')
    plt.ylabel('Density')
    plt.title(f'Distribution of Sample Norms (n={n}, q={q}, σ={sigma})')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'ntru_sample_norms.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved norm distribution plot to results/figures/")
    
    # Plot first two coordinates
    plt.figure(figsize=(8, 8))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10)
    plt.xlabel('Coordinate 1')
    plt.ylabel('Coordinate 2')
    plt.title(f'First Two Coordinates of NTRU Lattice Samples (n={n})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(os.path.join(plot_dir, 'ntru_sample_2d.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved 2D projection plot to results/figures/")
    
    return True


def test_ntru_cvp_accuracy():
    """Test accuracy of CVP solver on NTRU lattice."""
    print("\n\nTesting CVP accuracy on NTRU lattice...")
    
    n = 32
    q = 257
    lattice = NTRULattice(n=n, q=q)
    
    # Generate keys
    keys = lattice.generate_keys()
    if keys is None:
        print("❌ Key generation failed")
        return False
    
    print("✓ Keys generated")
    
    # Test CVP with known lattice points
    print("\nTesting CVP with perturbed lattice points...")
    
    errors = []
    num_tests = 100
    max_perturbation = 0.5
    
    for i in range(num_tests):
        # Get a random lattice point
        coeffs = [randint(-5, 5) for _ in range(2*n)]
        v = lattice.basis_matrix * vector(coeffs)
        
        # Add small perturbation
        perturbation = vector([max_perturbation * (2*random() - 1) for _ in range(2*n)])
        target = v + perturbation
        
        # Solve CVP
        closest = lattice.closest_vector(target)
        
        # Check if we recovered the correct point
        if closest == v:
            errors.append(0)
        else:
            error = (closest - v).norm()
            errors.append(error)
    
    success_rate = sum(1 for e in errors if e == 0) / num_tests
    print(f"\nCVP Success rate: {success_rate*100:.1f}%")
    print(f"Average error when failed: {np.mean([e for e in errors if e > 0]):.4f}")
    
    return True


def test_performance_comparison():
    """Compare performance for different parameter sizes."""
    print("\n\nPerformance comparison for different parameters...")
    
    import time
    
    params = [
        (32, 257, "Small"),
        (64, 257, "Medium"),
        (128, 257, "Large"),
    ]
    
    results = []
    
    for n, q, label in params:
        print(f"\nTesting {label}: n={n}, q={q}")
        
        # Time key generation
        start = time.time()
        lattice = NTRULattice(n=n, q=q)
        keys = lattice.generate_keys(max_attempts=3)
        keygen_time = time.time() - start
        
        if keys is None:
            print(f"  Key generation failed")
            continue
        
        print(f"  Key generation: {keygen_time:.3f}s")
        
        # Time Gram-Schmidt
        start = time.time()
        gs_norms = lattice.gram_schmidt_norms()
        gs_time = time.time() - start
        print(f"  Gram-Schmidt: {gs_time:.3f}s")
        
        # Time CVP solving
        num_cvp = 100
        start = time.time()
        for _ in range(num_cvp):
            target = vector([normalvariate(0, 1) for _ in range(2*n)])
            lattice.closest_vector(target)
        cvp_time = (time.time() - start) / num_cvp
        print(f"  CVP (per call): {cvp_time*1000:.3f}ms")
        
        results.append({
            'n': n,
            'q': q,
            'label': label,
            'keygen_time': keygen_time,
            'gs_time': gs_time,
            'cvp_time': cvp_time
        })
    
    # Save results
    import json
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'ntru_performance.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved performance results to results/tables/")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("NTRU Lattice Integration Tests")
    print("=" * 60)
    
    # Run all integration tests
    success = True
    
    success &= test_ntru_gaussian_sampling()
    success &= test_ntru_cvp_accuracy()
    success &= test_performance_comparison()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)