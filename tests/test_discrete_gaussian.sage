#!/usr/bin/env sage
"""
Comprehensive unit tests for discrete Gaussian samplers.

Tests statistical properties, edge cases, and integration with lattices.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import *
from src.core.discrete_gaussian import (
    RejectionSampler, CDTSampler, DiscreteGaussianVectorSampler,
    sample_discrete_gaussian_1d, sample_discrete_gaussian_vec,
    LatticeGaussianSampler
)
import numpy as np
from scipy import stats


class TestDiscreteGaussian:
    """Test suite for discrete Gaussian samplers."""
    
    def __init__(self):
        self.verbose = True
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def test_rejection_sampler_basic(self):
        """Test basic functionality of rejection sampler."""
        self.log("\n=== Testing Rejection Sampler Basic ===")
        
        # Test initialization
        sampler = RejectionSampler(sigma=2.0, center=0.0)
        assert sampler.sigma == 2.0
        assert sampler.center == 0.0
        
        # Test single sample
        x = sampler.sample()
        assert isinstance(x, Integer)
        self.log(f"Single sample: {x}")
        
        # Test multiple samples
        samples = sampler.sample(100)
        assert len(samples) == 100
        assert all(isinstance(x, Integer) for x in samples)
        
        self.log("✓ Basic rejection sampler tests passed")
    
    def test_rejection_sampler_statistics(self):
        """Test statistical properties of rejection sampler."""
        self.log("\n=== Testing Rejection Sampler Statistics ===")
        
        # Test different parameters
        test_cases = [
            (1.0, 0.0),    # σ=1, c=0
            (2.5, 0.0),    # σ=2.5, c=0
            (1.5, 3.2),    # σ=1.5, c=3.2
            (5.0, -2.7),   # σ=5, c=-2.7
        ]
        
        for sigma, center in test_cases:
            self.log(f"\nTesting σ={sigma}, c={center}")
            sampler = RejectionSampler(sigma=sigma, center=center)
            
            # Generate samples
            n_samples = 10000
            samples = sampler.sample(n_samples)
            samples_float = [float(x) for x in samples]
            
            # Check empirical mean
            emp_mean = mean(samples_float)
            # For discrete Gaussian, mean ≈ center (within statistical error)
            mean_error = abs(emp_mean - center)
            mean_tolerance = 3 * sigma / sqrt(n_samples)  # 3 standard errors
            
            self.log(f"  Empirical mean: {emp_mean:.4f} (expected: {center})")
            self.log(f"  Mean error: {mean_error:.4f} (tolerance: {mean_tolerance:.4f})")
            assert mean_error < mean_tolerance, f"Mean error {mean_error} exceeds tolerance"
            
            # Check empirical variance
            emp_var = variance(samples_float)
            # For discrete Gaussian, variance ≈ σ² (approximately)
            self.log(f"  Empirical variance: {emp_var:.4f} (expected: ~{sigma**2:.4f})")
            
            # Chi-squared test for distribution
            if sigma < 3:  # Only for small sigma (manageable support)
                self._chi_squared_test(samples, sampler)
        
        self.log("\n✓ Statistical tests passed")
    
    def test_cdt_sampler(self):
        """Test CDT sampler functionality."""
        self.log("\n=== Testing CDT Sampler ===")
        
        # Test small sigma (should use CDT)
        sampler = CDTSampler(sigma=1.5, center=0.0)
        assert not sampler.use_rejection
        
        # Test probability computation
        p0 = sampler.probability(0)
        p1 = sampler.probability(1)
        p2 = sampler.probability(2)
        
        self.log(f"P(X=0) = {p0:.6f}")
        self.log(f"P(X=1) = {p1:.6f}")
        self.log(f"P(X=2) = {p2:.6f}")
        
        # Probabilities should decrease from center
        assert p0 > p1 > p2
        
        # Test sampling
        samples = sampler.sample(1000)
        assert len(samples) == 1000
        
        # Test large sigma (should fall back to rejection)
        large_sampler = CDTSampler(sigma=100.0, center=0.0, max_table_size=1000)
        assert large_sampler.use_rejection
        
        self.log("✓ CDT sampler tests passed")
    
    def test_vector_sampler(self):
        """Test vector discrete Gaussian sampler."""
        self.log("\n=== Testing Vector Sampler ===")
        
        # Test uniform sigma
        n = 5
        sigma = 2.0
        sampler = DiscreteGaussianVectorSampler(sigma=sigma, n=n)
        
        v = sampler.sample()
        assert len(v) == n
        assert all(x in ZZ for x in v)
        self.log(f"Sample vector: {v}")
        
        # Test with center
        center = vector([1.5, -0.5, 2.1, 0.0, -1.2])
        sampler_centered = DiscreteGaussianVectorSampler(
            sigma=sigma, center=center, n=n
        )
        
        # Generate samples and check mean
        samples = sampler_centered.sample(1000)
        emp_mean = vector(RDF, [mean([float(v[i]) for v in samples]) 
                                for i in range(n)])
        
        self.log(f"Empirical mean: {emp_mean}")
        self.log(f"Expected center: {center}")
        
        # Check each coordinate
        for i in range(n):
            error = abs(emp_mean[i] - center[i])
            assert error < 0.2, f"Coordinate {i} error too large"
        
        # Test non-uniform sigmas
        sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
        sampler_nonuniform = DiscreteGaussianVectorSampler(sigma=sigmas)
        v = sampler_nonuniform.sample()
        assert len(v) == len(sigmas)
        
        self.log("✓ Vector sampler tests passed")
    
    def test_edge_cases(self):
        """Test edge cases and extreme parameters."""
        self.log("\n=== Testing Edge Cases ===")
        
        # Very small sigma
        small_sampler = RejectionSampler(sigma=0.1, center=0.0)
        samples = small_sampler.sample(100)
        # Should mostly be 0
        zero_count = sum(1 for x in samples if x == 0)
        self.log(f"Small σ=0.1: {zero_count}/100 zeros")
        assert zero_count > 90
        
        # Very large sigma
        large_sampler = RejectionSampler(sigma=1000.0, center=0.0)
        x = large_sampler.sample()
        self.log(f"Large σ=1000 sample: {x}")
        
        # Extreme center
        extreme_sampler = RejectionSampler(sigma=2.0, center=1000.5)
        samples = extreme_sampler.sample(10)
        emp_mean = float(mean(samples))
        self.log(f"Extreme center 1000.5, empirical mean: {emp_mean}")
        assert 998 < emp_mean < 1003
        
        # Test invalid parameters
        try:
            RejectionSampler(sigma=-1.0)
            assert False, "Should raise error for negative sigma"
        except ValueError:
            self.log("✓ Correctly rejected negative sigma")
        
        self.log("✓ Edge case tests passed")
    
    def test_tail_probabilities(self):
        """Test tail probability bounds."""
        self.log("\n=== Testing Tail Probabilities ===")
        
        sigma = 2.0
        sampler = RejectionSampler(sigma=sigma, center=0.0)
        
        # Generate many samples
        n_samples = 100000
        samples = sampler.sample(n_samples)
        
        # Check tail bounds at different thresholds
        for k in [2, 3, 4, 5]:
            threshold = k * sigma
            tail_count = sum(1 for x in samples if abs(float(x)) > threshold)
            tail_prob = tail_count / n_samples
            
            # Gaussian tail bound (approximate for discrete)
            gaussian_bound = 2 * (1 - stats.norm.cdf(k))
            
            self.log(f"P(|X| > {k}σ) = {tail_prob:.6f} "
                    f"(Gaussian bound: {gaussian_bound:.6f})")
            
            # Should be same order of magnitude
            assert tail_prob < 2 * gaussian_bound
        
        self.log("✓ Tail probability tests passed")
    
    def test_lattice_integration(self):
        """Test integration with lattice structures."""
        self.log("\n=== Testing Lattice Integration ===")
        
        # Create simple 2D lattice
        B = matrix(ZZ, [[2, 1], [0, 2]])
        lattice_sampler = LatticeGaussianSampler(B, sigma=1.0)
        
        # Test nearest plane
        target = vector(RDF, [3.2, 2.7])
        nearest = lattice_sampler.nearest_plane(target)
        
        self.log(f"Target: {target}")
        self.log(f"Nearest lattice point: {nearest}")
        
        # Verify it's in the lattice
        coeffs = B.solve_left(nearest)
        assert all(c in ZZ for c in coeffs)
        
        self.log("✓ Lattice integration tests passed")
    
    def test_convenience_functions(self):
        """Test convenience sampling functions."""
        self.log("\n=== Testing Convenience Functions ===")
        
        # Test 1D sampling
        x = sample_discrete_gaussian_1d(sigma=2.0, center=1.0)
        assert x in ZZ
        self.log(f"1D sample: {x}")
        
        # Test vector sampling
        v = sample_discrete_gaussian_vec(sigma=1.5, n=5)
        assert len(v) == 5
        assert v.parent() == ZZ^5
        self.log(f"Vector sample: {v}")
        
        # Test with per-coordinate sigmas
        v2 = sample_discrete_gaussian_vec(
            sigma=[1.0, 1.5, 2.0], 
            center=[0, 1, -1]
        )
        assert len(v2) == 3
        self.log(f"Non-uniform sample: {v2}")
        
        self.log("✓ Convenience function tests passed")
    
    def test_performance(self):
        """Test performance for cryptographic parameters."""
        self.log("\n=== Testing Performance ===")
        
        import time
        
        # Test rejection sampler performance
        sigma = 1.17 * sqrt(12289)  # FALCON parameter
        sampler = RejectionSampler(sigma=sigma)
        
        start = time.time()
        samples = sampler.sample(10000)
        elapsed = time.time() - start
        
        rate = 10000 / elapsed
        self.log(f"Rejection sampler: {rate:.0f} samples/sec (σ={sigma:.2f})")
        
        # Test CDT sampler performance
        cdt_sampler = CDTSampler(sigma=5.0)
        
        start = time.time()
        samples = cdt_sampler.sample(10000)
        elapsed = time.time() - start
        
        rate = 10000 / elapsed
        self.log(f"CDT sampler: {rate:.0f} samples/sec (σ=5.0)")
        
        # Test vector sampler performance
        n = 512  # FALCON-512 dimension
        vec_sampler = DiscreteGaussianVectorSampler(sigma=sigma, n=n)
        
        start = time.time()
        v = vec_sampler.sample()
        elapsed = time.time() - start
        
        self.log(f"Vector sample (n={n}): {elapsed*1000:.2f} ms")
        
        self.log("✓ Performance tests completed")
    
    def _chi_squared_test(self, samples, sampler):
        """Perform chi-squared goodness of fit test."""
        from collections import Counter
        
        # Count occurrences
        counts = Counter(samples)
        
        # Get theoretical probabilities for observed support
        observed_values = sorted(counts.keys())
        
        # Compute expected frequencies
        n_samples = len(samples)
        expected = []
        observed = []
        
        for x in observed_values:
            exp_prob = sampler.probability(x)
            exp_count = n_samples * exp_prob
            
            # Only include if expected count > 5 (chi-squared requirement)
            if exp_count > 5:
                expected.append(exp_count)
                observed.append(counts[x])
        
        if len(expected) < 2:
            return  # Not enough data for test
        
        # Compute chi-squared statistic
        chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
        df = len(expected) - 1
        
        # Get p-value
        p_value = 1 - RDF(stats.chi2.cdf(float(chi2), df))
        
        self.log(f"  Chi-squared test: χ²={chi2:.2f}, df={df}, p={p_value:.4f}")
        
        # We use a lenient threshold since discrete != continuous
        assert p_value > 0.01, f"Chi-squared test failed (p={p_value})"


def run_all_tests():
    """Run all discrete Gaussian tests."""
    print("Running Discrete Gaussian Sampler Tests")
    print("=" * 60)
    
    tester = TestDiscreteGaussian()
    
    # Run all test methods
    tester.test_rejection_sampler_basic()
    tester.test_rejection_sampler_statistics()
    tester.test_cdt_sampler()
    tester.test_vector_sampler()
    tester.test_edge_cases()
    tester.test_tail_probabilities()
    tester.test_lattice_integration()
    tester.test_convenience_functions()
    tester.test_performance()
    
    print("\n" + "=" * 60)
    print("✅ All discrete Gaussian tests passed!")


if __name__ == "__main__":
    run_all_tests()