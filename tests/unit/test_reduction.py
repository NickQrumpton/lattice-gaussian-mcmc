"""
Unit tests for lattice reduction algorithms.

Tests cover correctness, numerical accuracy, quality metrics, and performance
for LLL and BKZ reduction algorithms.
"""

import numpy as np
import pytest
from typing import Tuple, List
from unittest.mock import Mock, patch
import time

from lattices.reduction import LatticeReduction
from lattices.identity import IdentityLattice
from lattices.qary import QaryLattice


class TestLLLReduction:
    """Test LLL (Lenstra-Lenstra-Lovász) reduction algorithm."""
    
    def test_lll_basic_functionality(self, random_2d_basis, tolerance_config):
        """Test basic LLL reduction functionality."""
        reducer = LatticeReduction()
        
        # Perform LLL reduction
        reduced_basis, unimodular = reducer.lll_reduce(random_2d_basis)
        
        # Check dimensions are preserved
        assert reduced_basis.shape == random_2d_basis.shape
        assert unimodular.shape == (random_2d_basis.shape[0], random_2d_basis.shape[0])
        
        # Check unimodular transformation
        reconstructed = unimodular @ random_2d_basis
        np.testing.assert_allclose(
            reconstructed, reduced_basis, 
            rtol=tolerance_config['rtol'],
            err_msg="Unimodular transformation failed"
        )
        
        # Check determinant preservation
        original_det = np.abs(np.linalg.det(random_2d_basis @ random_2d_basis.T))
        reduced_det = np.abs(np.linalg.det(reduced_basis @ reduced_basis.T))
        
        np.testing.assert_allclose(
            original_det, reduced_det,
            rtol=tolerance_config['rtol'],
            err_msg="LLL reduction changed lattice determinant"
        )
    
    def test_lll_lovasz_condition(self, random_2d_basis, tolerance_config):
        """Test that LLL satisfies the Lovász condition."""
        reducer = LatticeReduction()
        reduced_basis, _ = reducer.lll_reduce(random_2d_basis, delta=0.75)
        
        # Compute Gram-Schmidt orthogonalization
        Q, R = np.linalg.qr(reduced_basis.T)
        gram_schmidt_norms = np.abs(np.diag(R))
        
        # Check Lovász condition: ||μ_{i,j}|| ≤ 1/2 for i > j
        # and δ||b*_i||² ≤ ||b*_{i+1} + μ_{i+1,i}b*_i||²
        
        if len(gram_schmidt_norms) > 1:
            for i in range(len(gram_schmidt_norms) - 1):
                # Simplified Lovász condition check
                ratio = gram_schmidt_norms[i+1] / gram_schmidt_norms[i]
                assert ratio >= 0.5, f"Lovász condition violated at index {i}: ratio = {ratio}"
    
    def test_lll_size_reduction(self, random_2d_basis, tolerance_config):
        """Test that LLL produces size-reduced basis."""
        reducer = LatticeReduction()
        reduced_basis, _ = reducer.lll_reduce(random_2d_basis)
        
        # Compute Gram-Schmidt coefficients
        Q, R = np.linalg.qr(reduced_basis.T)
        
        # Check size reduction: |μ_{i,j}| ≤ 1/2 for i > j
        n = R.shape[0]
        for i in range(n):
            for j in range(i):
                if abs(R[j, j]) > tolerance_config['atol']:
                    mu_ij = R[j, i] / R[j, j]
                    assert abs(mu_ij) <= 0.5 + tolerance_config['rtol'], \
                        f"Size reduction violated: μ_{i},{j} = {mu_ij}"
    
    @pytest.mark.numerical
    def test_lll_quality_improvement(self, pathological_basis):
        """Test that LLL improves basis quality metrics."""
        reducer = LatticeReduction()
        
        # Compute quality metrics before reduction
        original_condition = np.linalg.cond(pathological_basis)
        original_orthogonality = reducer.orthogonality_defect(pathological_basis)
        
        # Perform reduction
        reduced_basis, _ = reducer.lll_reduce(pathological_basis)
        
        # Compute quality metrics after reduction
        reduced_condition = np.linalg.cond(reduced_basis)
        reduced_orthogonality = reducer.orthogonality_defect(reduced_basis)
        
        # Quality should improve (lower is better)
        assert reduced_condition <= original_condition * 2, \
            "LLL failed to improve condition number significantly"
        assert reduced_orthogonality <= original_orthogonality, \
            "LLL failed to improve orthogonality defect"
    
    @pytest.mark.edge_case
    def test_lll_identity_matrix(self):
        """Test LLL on identity matrix (should remain unchanged)."""
        reducer = LatticeReduction()
        identity = np.eye(3)
        
        reduced_basis, unimodular = reducer.lll_reduce(identity)
        
        # Should remain identity (up to reordering)
        np.testing.assert_allclose(reduced_basis, identity, rtol=1e-10)
        np.testing.assert_allclose(unimodular, np.eye(3), rtol=1e-10)
    
    @pytest.mark.edge_case
    def test_lll_singular_matrix(self):
        """Test LLL with singular/rank-deficient matrix."""
        reducer = LatticeReduction()
        
        # Create rank-deficient matrix
        singular_basis = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            reducer.lll_reduce(singular_basis)
    
    def test_lll_different_deltas(self, random_2d_basis):
        """Test LLL with different δ parameters."""
        reducer = LatticeReduction()
        deltas = [0.5, 0.75, 0.99]
        
        previous_quality = float('inf')
        
        for delta in deltas:
            reduced_basis, _ = reducer.lll_reduce(random_2d_basis, delta=delta)
            quality = reducer.orthogonality_defect(reduced_basis)
            
            # Higher delta should give better quality (generally)
            if delta > 0.5:
                assert quality <= previous_quality * 1.5, \
                    f"Quality degraded unexpectedly for delta={delta}"
            previous_quality = quality
    
    @pytest.mark.slow
    def test_lll_performance(self, performance_config):
        """Test LLL reduction performance."""
        reducer = LatticeReduction()
        
        # Test on different matrix sizes
        dimensions = [5, 10, 15, 20]
        
        for dim in dimensions:
            # Create random well-conditioned basis
            A = np.random.randn(dim, dim)
            U, s, Vt = np.linalg.svd(A)
            # Set condition number to be reasonable
            s = np.linspace(1, 10, dim)
            basis = U @ np.diag(s) @ Vt
            
            start_time = time.time()
            reduced_basis, _ = reducer.lll_reduce(basis)
            elapsed_time = time.time() - start_time
            
            # Performance should be reasonable
            max_time = performance_config['max_time_reduction'] * (dim / 20)**2
            assert elapsed_time < max_time, \
                f"LLL too slow for dimension {dim}: {elapsed_time:.2f}s > {max_time:.2f}s"


class TestBKZReduction:
    """Test BKZ (Block Korkine-Zolotarev) reduction algorithm."""
    
    def test_bkz_basic_functionality(self, random_2d_basis, tolerance_config):
        """Test basic BKZ reduction functionality."""
        reducer = LatticeReduction()
        
        # Perform BKZ reduction with small block size
        reduced_basis, unimodular = reducer.bkz_reduce(random_2d_basis, block_size=2)
        
        # Check dimensions are preserved
        assert reduced_basis.shape == random_2d_basis.shape
        assert unimodular.shape == (random_2d_basis.shape[0], random_2d_basis.shape[0])
        
        # Check unimodular transformation
        reconstructed = unimodular @ random_2d_basis
        np.testing.assert_allclose(
            reconstructed, reduced_basis,
            rtol=tolerance_config['rtol'],
            err_msg="BKZ unimodular transformation failed"
        )
    
    def test_bkz_quality_improvement(self, pathological_basis):
        """Test that BKZ improves basis quality."""
        reducer = LatticeReduction()
        
        # Compute quality before reduction
        original_orthogonality = reducer.orthogonality_defect(pathological_basis)
        original_hermite = reducer.hermite_factor(pathological_basis)
        
        # Perform BKZ reduction
        reduced_basis, _ = reducer.bkz_reduce(pathological_basis, block_size=2)
        
        # Compute quality after reduction
        reduced_orthogonality = reducer.orthogonality_defect(reduced_basis)
        reduced_hermite = reducer.hermite_factor(reduced_basis)
        
        # Quality should improve or stay same
        assert reduced_orthogonality <= original_orthogonality * 1.1, \
            "BKZ failed to maintain orthogonality defect"
        assert reduced_hermite <= original_hermite * 1.1, \
            "BKZ failed to maintain Hermite factor"
    
    def test_bkz_different_block_sizes(self, standard_3d_basis):
        """Test BKZ with different block sizes."""
        reducer = LatticeReduction()
        
        # Make basis non-trivial
        transform = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
        basis = transform @ standard_3d_basis
        
        block_sizes = [2, 3]
        previous_quality = float('inf')
        
        for block_size in block_sizes:
            reduced_basis, _ = reducer.bkz_reduce(basis, block_size=block_size)
            quality = reducer.orthogonality_defect(reduced_basis)
            
            # Larger block size should generally give better quality
            assert quality <= previous_quality * 1.2, \
                f"BKZ quality degraded for block_size={block_size}"
            previous_quality = quality
    
    @pytest.mark.edge_case
    def test_bkz_invalid_block_size(self, random_2d_basis):
        """Test BKZ with invalid block sizes."""
        reducer = LatticeReduction()
        
        # Block size too small
        with pytest.raises(ValueError, match="Block size must be at least 2"):
            reducer.bkz_reduce(random_2d_basis, block_size=1)
        
        # Block size too large
        with pytest.raises(ValueError, match="Block size cannot exceed"):
            reducer.bkz_reduce(random_2d_basis, block_size=10)
    
    @pytest.mark.slow
    def test_bkz_convergence(self, standard_3d_basis):
        """Test BKZ convergence with multiple tours."""
        reducer = LatticeReduction()
        
        # Create non-trivial basis
        transform = np.random.randint(-5, 6, (3, 3))
        while np.abs(np.linalg.det(transform)) != 1:
            transform = np.random.randint(-5, 6, (3, 3))
        basis = transform.astype(float) @ standard_3d_basis
        
        # Run BKZ with multiple tours
        reduced_basis, _ = reducer.bkz_reduce(basis, block_size=2, max_tours=5)
        
        # Should converge to a reasonable basis
        orthogonality = reducer.orthogonality_defect(reduced_basis)
        assert orthogonality < 10, "BKZ failed to achieve reasonable orthogonality"


class TestReductionQualityMetrics:
    """Test quality metrics for lattice reduction."""
    
    def test_orthogonality_defect_computation(self, tolerance_config):
        """Test orthogonality defect computation."""
        reducer = LatticeReduction()
        
        # Test on orthogonal matrix (should have defect = 1)
        orthogonal_basis = np.array([[1.0, 0.0], [0.0, 1.0]])
        defect_orth = reducer.orthogonality_defect(orthogonal_basis)
        
        np.testing.assert_allclose(
            defect_orth, 1.0, rtol=tolerance_config['rtol'],
            err_msg="Orthogonality defect incorrect for orthogonal basis"
        )
        
        # Test on pathological basis (should have large defect)
        pathological = np.array([[1.0, 0.0], [1e-6, 1.0]])
        defect_path = reducer.orthogonality_defect(pathological)
        
        assert defect_path > 1.0, "Pathological basis should have defect > 1"
        assert defect_path < 1e12, "Orthogonality defect seems too large"
    
    def test_hermite_factor_computation(self, tolerance_config):
        """Test Hermite factor computation."""
        reducer = LatticeReduction()
        
        # Test on identity lattice
        identity = np.eye(2)
        hermite = reducer.hermite_factor(identity)
        
        # For 2D identity lattice, Hermite factor should be 1
        np.testing.assert_allclose(
            hermite, 1.0, rtol=tolerance_config['rtol'],
            err_msg="Hermite factor incorrect for identity lattice"
        )
        
        # Test that Hermite factor is always ≥ 1
        random_basis = np.random.randn(3, 3)
        hermite_random = reducer.hermite_factor(random_basis)
        
        assert hermite_random >= 1.0 - tolerance_config['rtol'], \
            "Hermite factor must be ≥ 1"
    
    def test_shortest_vector_approximation(self, tolerance_config):
        """Test shortest vector approximation."""
        reducer = LatticeReduction()
        
        # For identity lattice, shortest vector should be (1,0) or (0,1)
        identity = np.eye(2)
        shortest = reducer.approximate_shortest_vector(identity)
        
        expected_length = 1.0
        actual_length = np.linalg.norm(shortest)
        
        np.testing.assert_allclose(
            actual_length, expected_length, rtol=tolerance_config['rtol'],
            err_msg="Shortest vector approximation incorrect for identity lattice"
        )
    
    def test_successive_minima_bounds(self, standard_3d_basis):
        """Test successive minima computation."""
        reducer = LatticeReduction()
        
        # Compute successive minima
        minima = reducer.successive_minima(standard_3d_basis)
        
        # Should have correct number of minima
        assert len(minima) == 3, "Wrong number of successive minima"
        
        # Should be in non-decreasing order
        for i in range(len(minima) - 1):
            assert minima[i] <= minima[i+1] * (1 + 1e-10), \
                "Successive minima not in non-decreasing order"
        
        # All should be positive
        assert all(m > 0 for m in minima), "All minima must be positive"


class TestSamplingOrientedReduction:
    """Test reduction algorithms optimized for sampling."""
    
    def test_sampling_reduce_basic(self, random_2d_basis, tolerance_config):
        """Test sampling-oriented reduction."""
        reducer = LatticeReduction()
        sigma = 1.0
        
        reduced_basis = reducer.sampling_reduce(random_2d_basis, sigma)
        
        # Check dimensions preserved
        assert reduced_basis.shape == random_2d_basis.shape
        
        # Check lattice preserved (same determinant)
        original_det = np.abs(np.linalg.det(random_2d_basis @ random_2d_basis.T))
        reduced_det = np.abs(np.linalg.det(reduced_basis @ reduced_basis.T))
        
        np.testing.assert_allclose(
            original_det, reduced_det, rtol=tolerance_config['rtol'],
            err_msg="Sampling reduction changed lattice determinant"
        )
    
    def test_sampling_reduce_quality(self, pathological_basis):
        """Test that sampling reduction improves sampling quality."""
        reducer = LatticeReduction()
        sigma = 1.0
        
        # Compute sampling quality before reduction
        original_quality = reducer.sampling_quality_metric(pathological_basis, sigma)
        
        # Perform sampling-oriented reduction
        reduced_basis = reducer.sampling_reduce(pathological_basis, sigma)
        
        # Compute sampling quality after reduction
        reduced_quality = reducer.sampling_quality_metric(reduced_basis, sigma)
        
        # Quality should improve (lower is better for this metric)
        assert reduced_quality <= original_quality * 1.1, \
            "Sampling reduction failed to improve sampling quality"
    
    def test_sampling_quality_metric(self, identity_lattice_2d):
        """Test sampling quality metric computation."""
        reducer = LatticeReduction()
        basis = identity_lattice_2d.get_basis()
        
        # Test with different sigma values
        sigmas = [0.1, 1.0, 10.0]
        
        for sigma in sigmas:
            quality = reducer.sampling_quality_metric(basis, sigma)
            
            assert quality > 0, f"Sampling quality metric must be positive for sigma={sigma}"
            assert np.isfinite(quality), f"Sampling quality metric must be finite for sigma={sigma}"
    
    def test_adaptive_reduction_strategy(self, random_2d_basis):
        """Test adaptive reduction strategy selection."""
        reducer = LatticeReduction()
        
        # Test different conditioning scenarios
        strategies = reducer.select_reduction_strategy(random_2d_basis)
        
        assert isinstance(strategies, list), "Reduction strategies should be a list"
        assert len(strategies) > 0, "Should recommend at least one strategy"
        assert all(isinstance(s, str) for s in strategies), "Strategies should be strings"


@pytest.mark.integration
class TestReductionIntegration:
    """Integration tests for reduction with lattices and samplers."""
    
    def test_reduction_with_qary_lattice(self, qary_lattice_small):
        """Test reduction algorithms on q-ary lattice."""
        reducer = LatticeReduction()
        original_basis = qary_lattice_small.get_basis()
        
        # Test LLL reduction
        reduced_lll, _ = reducer.lll_reduce(original_basis)
        
        # Should improve quality
        original_orthogonality = reducer.orthogonality_defect(original_basis)
        reduced_orthogonality = reducer.orthogonality_defect(reduced_lll)
        
        assert reduced_orthogonality <= original_orthogonality * 1.1, \
            "LLL reduction failed to improve q-ary lattice"
    
    def test_reduction_for_sampling_pipeline(self, pathological_basis):
        """Test reduction as preprocessing for sampling."""
        from samplers.klein import KleinSampler
        
        reducer = LatticeReduction()
        sigma = 1.0
        
        # Create mock lattice with pathological basis
        class TestLattice:
            def __init__(self, basis):
                self.basis = basis
                
            def get_basis(self):
                return self.basis
                
            def get_gram_schmidt(self):
                Q, R = np.linalg.qr(self.basis.T)
                return Q.T, R.T
                
            def decode_cvp(self, target):
                return np.round(target)
                
            def sample_lattice_point(self, sigma):
                return np.zeros(self.get_dimension())
                
            def get_dimension(self):
                return self.basis.shape[0]
        
        # Test sampling before reduction
        original_lattice = TestLattice(pathological_basis)
        
        try:
            original_sampler = KleinSampler(original_lattice, sigma)
            original_sample = original_sampler.sample()
        except (np.linalg.LinAlgError, ValueError):
            original_sample = None  # Sampling failed
        
        # Reduce basis and test sampling
        reduced_basis = reducer.sampling_reduce(pathological_basis, sigma)
        reduced_lattice = TestLattice(reduced_basis)
        
        reduced_sampler = KleinSampler(reduced_lattice, sigma)
        reduced_sample = reduced_sampler.sample()
        
        # Should successfully produce a sample
        assert reduced_sample is not None, "Sampling failed even after reduction"
        assert len(reduced_sample) == 2, "Sample has wrong dimension"
        assert all(np.isfinite(reduced_sample)), "Sample contains non-finite values"
    
    @pytest.mark.slow
    def test_reduction_performance_comparison(self, performance_config):
        """Compare performance of different reduction algorithms."""
        reducer = LatticeReduction()
        
        # Create test basis
        dim = 10
        basis = np.random.randn(dim, dim)
        basis = basis @ basis.T  # Make positive definite
        
        # Time LLL reduction
        start_time = time.time()
        reduced_lll, _ = reducer.lll_reduce(basis)
        lll_time = time.time() - start_time
        
        # Time BKZ reduction (smaller block size for speed)
        start_time = time.time()
        reduced_bkz, _ = reducer.bkz_reduce(basis, block_size=2, max_tours=1)
        bkz_time = time.time() - start_time
        
        # Compare quality
        lll_quality = reducer.orthogonality_defect(reduced_lll)
        bkz_quality = reducer.orthogonality_defect(reduced_bkz)
        
        # BKZ should give better or equal quality (but take more time)
        assert bkz_quality <= lll_quality * 1.1, \
            "BKZ should give better quality than LLL"
        
        # Both should complete in reasonable time
        max_time = performance_config['max_time_reduction']
        assert lll_time < max_time, f"LLL too slow: {lll_time:.2f}s"
        assert bkz_time < max_time * 2, f"BKZ too slow: {bkz_time:.2f}s"


@pytest.mark.reproducibility
class TestReductionReproducibility:
    """Test reproducibility of reduction algorithms."""
    
    def test_lll_reproducibility(self, random_2d_basis, test_seed):
        """Test that LLL reduction is reproducible."""
        reducer = LatticeReduction()
        
        # Run LLL twice with same input
        np.random.seed(test_seed)
        reduced1, unimodular1 = reducer.lll_reduce(random_2d_basis.copy())
        
        np.random.seed(test_seed)
        reduced2, unimodular2 = reducer.lll_reduce(random_2d_basis.copy())
        
        # Results should be identical
        np.testing.assert_array_almost_equal(reduced1, reduced2, decimal=10)
        np.testing.assert_array_almost_equal(unimodular1, unimodular2, decimal=10)
    
    def test_bkz_reproducibility(self, standard_3d_basis, test_seed):
        """Test that BKZ reduction is reproducible."""
        reducer = LatticeReduction()
        
        # Make basis non-trivial
        transform = np.array([[1, 2, 1], [0, 1, 1], [0, 0, 1]])
        basis = transform @ standard_3d_basis
        
        # Run BKZ twice with same input and seed
        np.random.seed(test_seed)
        reduced1, unimodular1 = reducer.bkz_reduce(basis.copy(), block_size=2)
        
        np.random.seed(test_seed)  
        reduced2, unimodular2 = reducer.bkz_reduce(basis.copy(), block_size=2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(reduced1, reduced2, decimal=10)
        np.testing.assert_array_almost_equal(unimodular1, unimodular2, decimal=10)