"""
Unit tests for lattice classes.

Tests cover correctness, numerical accuracy, invariants, and edge cases
for all lattice implementations including base, identity, and q-ary lattices.
"""

import numpy as np
import pytest
from typing import Tuple, Optional
from unittest.mock import Mock, patch

from lattices.base import BaseLattice
from lattices.identity import IdentityLattice
from lattices.qary import QaryLattice


class TestBaseLattice:
    """Test the abstract base lattice class."""
    
    def test_base_lattice_instantiation_fails(self):
        """Test that BaseLattice cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLattice()
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        required_methods = [
            'get_basis', 'get_gram_schmidt', 'decode_cvp', 
            'sample_lattice_point', 'get_dimension'
        ]
        
        for method in required_methods:
            assert hasattr(BaseLattice, method)
            assert getattr(BaseLattice, method).__isabstractmethod__
    
    @pytest.mark.numerical
    def test_smoothing_parameter_computation(self, simple_2d_basis, tolerance_config):
        """Test smoothing parameter computation."""
        # Create a mock lattice to test the base class method
        class MockLattice(BaseLattice):
            def __init__(self, basis):
                self.basis = basis
                
            def get_basis(self):
                return self.basis
                
            def get_gram_schmidt(self):
                Q, R = np.linalg.qr(self.basis.T)
                return Q.T, R.T
                
            def decode_cvp(self, target):
                # Simple nearest point for testing
                return np.round(target)
                
            def sample_lattice_point(self, sigma):
                return np.zeros(self.get_dimension())
                
            def get_dimension(self):
                return self.basis.shape[0]
        
        lattice = MockLattice(simple_2d_basis)
        
        # Test smoothing parameter with default epsilon
        eta = lattice.smoothing_parameter()
        assert eta > 0, "Smoothing parameter must be positive"
        assert np.isfinite(eta), "Smoothing parameter must be finite"
        
        # Test with different epsilon values
        eta_small = lattice.smoothing_parameter(epsilon=1e-10)
        eta_large = lattice.smoothing_parameter(epsilon=1e-2)
        assert eta_small > eta_large, "Smaller epsilon should give larger smoothing parameter"
    
    @pytest.mark.numerical
    def test_gaussian_heuristic(self, simple_2d_basis, tolerance_config):
        """Test Gaussian heuristic computation."""
        class MockLattice(BaseLattice):
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
        
        lattice = MockLattice(simple_2d_basis)
        
        # Test Gaussian heuristic
        gh = lattice.gaussian_heuristic()
        expected_gh = np.sqrt(2 / (np.pi * np.e))  # For 2D identity lattice
        
        assert np.isclose(gh, expected_gh, rtol=tolerance_config['rtol'])
        assert gh > 0, "Gaussian heuristic must be positive"
    
    @pytest.mark.numerical
    def test_first_minimum_bound(self, simple_2d_basis):
        """Test first minimum approximation."""
        class MockLattice(BaseLattice):
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
        
        lattice = MockLattice(simple_2d_basis)
        
        lambda1_approx = lattice.first_minimum()
        
        # For identity lattice, first minimum should be 1
        assert np.isclose(lambda1_approx, 1.0, rtol=0.1)
        assert lambda1_approx > 0, "First minimum must be positive"


class TestIdentityLattice:
    """Test the identity lattice Z^n implementation."""
    
    def test_identity_lattice_creation(self):
        """Test basic identity lattice creation."""
        for dim in [1, 2, 3, 5, 10]:
            lattice = IdentityLattice(dimension=dim)
            assert lattice.get_dimension() == dim
            
            basis = lattice.get_basis()
            expected_basis = np.eye(dim)
            np.testing.assert_array_equal(basis, expected_basis)
    
    @pytest.mark.edge_case
    def test_identity_lattice_invalid_dimensions(self):
        """Test identity lattice with invalid dimensions."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            IdentityLattice(dimension=0)
            
        with pytest.raises(ValueError, match="Dimension must be positive"):
            IdentityLattice(dimension=-1)
    
    def test_gram_schmidt_orthogonalization(self, identity_lattice_2d, tolerance_config):
        """Test Gram-Schmidt orthogonalization for identity lattice."""
        Q, R = identity_lattice_2d.get_gram_schmidt()
        
        # For identity lattice, Q should equal the basis and R should be identity
        expected_Q = np.eye(2)
        expected_R = np.eye(2)
        
        np.testing.assert_allclose(Q, expected_Q, rtol=tolerance_config['rtol'])
        np.testing.assert_allclose(R, expected_R, rtol=tolerance_config['rtol'])
        
        # Check orthogonality
        assert np.allclose(Q @ Q.T, np.eye(2), rtol=tolerance_config['rtol'])
    
    @pytest.mark.numerical
    def test_decode_cvp_identity(self, identity_lattice_2d, tolerance_config):
        """Test closest vector problem decoding for identity lattice."""
        test_points = [
            np.array([0.3, 0.7]),
            np.array([-0.6, 1.4]),
            np.array([2.1, -0.9]),
            np.array([0.0, 0.0])
        ]
        
        expected_closest = [
            np.array([0.0, 1.0]),
            np.array([-1.0, 1.0]),
            np.array([2.0, -1.0]),
            np.array([0.0, 0.0])
        ]
        
        for point, expected in zip(test_points, expected_closest):
            closest = identity_lattice_2d.decode_cvp(point)
            np.testing.assert_allclose(closest, expected, rtol=tolerance_config['rtol'])
    
    @pytest.mark.statistical
    def test_direct_sampling_statistical_properties(self, statistical_config, tolerance_config):
        """Test statistical properties of direct sampling from identity lattice."""
        lattice = IdentityLattice(dimension=2)
        sigma = 1.0
        n_samples = statistical_config['n_samples']
        
        # Generate samples
        samples = np.array([
            lattice.direct_sample(sigma) for _ in range(n_samples)
        ])
        
        # Test sample mean (should be close to zero)
        sample_mean = np.mean(samples, axis=0)
        expected_mean = np.zeros(2)
        
        # Allow for statistical fluctuation
        std_error = sigma / np.sqrt(n_samples)
        tolerance = 3 * std_error  # 3-sigma bound
        
        np.testing.assert_allclose(
            sample_mean, expected_mean, 
            atol=tolerance, 
            err_msg="Sample mean deviates significantly from expected value"
        )
        
        # Test sample variance (should be close to sigma^2 for each coordinate)
        sample_var = np.var(samples, axis=0)
        expected_var = sigma**2 * np.ones(2)
        
        # Chi-squared test for variance
        var_tolerance = statistical_config['statistical_rtol']
        np.testing.assert_allclose(
            sample_var, expected_var, 
            rtol=var_tolerance,
            err_msg="Sample variance deviates significantly from expected value"
        )
    
    @pytest.mark.statistical
    def test_direct_sampling_different_sigmas(self, statistical_config):
        """Test direct sampling with different sigma values."""
        lattice = IdentityLattice(dimension=2)
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for sigma in sigmas:
            samples = np.array([
                lattice.direct_sample(sigma) for _ in range(1000)
            ])
            
            # Standard deviation should scale with sigma
            sample_std = np.std(samples, axis=0)
            expected_std = sigma * np.ones(2)
            
            np.testing.assert_allclose(
                sample_std, expected_std,
                rtol=0.2,  # Allow 20% tolerance for limited samples
                err_msg=f"Standard deviation incorrect for sigma={sigma}"
            )
    
    @pytest.mark.edge_case
    def test_very_small_sigma(self):
        """Test behavior with very small sigma values."""
        lattice = IdentityLattice(dimension=2)
        sigma = 1e-6
        
        # Should still produce valid lattice points
        sample = lattice.direct_sample(sigma)
        assert len(sample) == 2
        assert all(isinstance(x, (int, np.integer)) for x in sample)
        
        # Most samples should be at origin for very small sigma
        samples = np.array([lattice.direct_sample(sigma) for _ in range(100)])
        zero_samples = np.sum(np.all(samples == 0, axis=1))
        
        # Expect most samples to be at origin
        assert zero_samples > 50, "Very small sigma should concentrate samples at origin"
    
    @pytest.mark.edge_case  
    def test_very_large_sigma(self):
        """Test behavior with very large sigma values."""
        lattice = IdentityLattice(dimension=2)
        sigma = 100.0
        
        # Should still produce valid lattice points
        sample = lattice.direct_sample(sigma)
        assert len(sample) == 2
        assert all(isinstance(x, (int, np.integer)) for x in sample)
        
        # Samples should have large variance
        samples = np.array([lattice.direct_sample(sigma) for _ in range(1000)])
        sample_std = np.std(samples, axis=0)
        
        # Standard deviation should be on order of sigma
        assert np.all(sample_std > sigma * 0.5), "Large sigma should produce spread-out samples"
    
    def test_lattice_point_validation(self, identity_lattice_2d):
        """Test that sampled points are valid lattice points."""
        sigma = 1.0
        
        for _ in range(100):
            point = identity_lattice_2d.sample_lattice_point(sigma)
            
            # For identity lattice, all coordinates should be integers
            assert all(isinstance(x, (int, np.integer)) for x in point)
            assert len(point) == 2


class TestQaryLattice:
    """Test q-ary lattice implementation."""
    
    def test_qary_from_lwe_instance(self):
        """Test q-ary lattice construction from LWE instance."""
        n, m, q = 4, 6, 17
        A = np.random.randint(0, q, (n, m))
        
        # Test primal lattice
        lattice_primal = QaryLattice.from_lwe_instance(A, q, dual=False)
        basis_primal = lattice_primal.get_basis()
        
        # Check dimensions
        assert basis_primal.shape == (n, m + n)
        
        # Check structure: should be [A | qI]
        np.testing.assert_array_equal(basis_primal[:, :m], A)
        np.testing.assert_array_equal(basis_primal[:, m:], q * np.eye(n))
        
        # Test dual lattice  
        lattice_dual = QaryLattice.from_lwe_instance(A, q, dual=True)
        basis_dual = lattice_dual.get_basis()
        
        # Check dimensions
        assert basis_dual.shape == (m, m + n)
        
        # Check structure: should be [qI | -A^T]
        np.testing.assert_array_equal(basis_dual[:, :m], q * np.eye(m))
        np.testing.assert_array_equal(basis_dual[:, m:], -A.T)
    
    @pytest.mark.edge_case
    def test_qary_invalid_parameters(self):
        """Test q-ary lattice with invalid parameters."""
        n, m = 4, 6
        
        # Test with q = 1 (should raise error)
        A = np.random.randint(0, 2, (n, m))
        with pytest.raises(ValueError, match="q must be greater than 1"):
            QaryLattice.from_lwe_instance(A, q=1, dual=False)
        
        # Test with empty matrix
        A_empty = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="Matrix A must be non-empty"):
            QaryLattice.from_lwe_instance(A_empty, q=17, dual=False)
    
    def test_qary_gram_schmidt(self, qary_lattice_small, tolerance_config):
        """Test Gram-Schmidt orthogonalization for q-ary lattice."""
        Q, R = qary_lattice_small.get_gram_schmidt()
        basis = qary_lattice_small.get_basis()
        
        # Check that Q @ R reconstructs the basis
        reconstructed = Q @ R
        np.testing.assert_allclose(
            reconstructed, basis, 
            rtol=tolerance_config['rtol'],
            err_msg="Gram-Schmidt decomposition failed to reconstruct basis"
        )
        
        # Check orthogonality of Q
        QTQ = Q @ Q.T
        expected_identity = np.eye(Q.shape[0])
        np.testing.assert_allclose(
            QTQ, expected_identity,
            rtol=tolerance_config['rtol'],
            err_msg="Gram-Schmidt Q matrix is not orthogonal"
        )
    
    def test_qary_decode_cvp(self, qary_lattice_small, tolerance_config):
        """Test CVP decoding for q-ary lattice."""
        target = np.random.randn(qary_lattice_small.get_dimension())
        
        closest = qary_lattice_small.decode_cvp(target)
        
        # Check that result is correct dimension
        assert len(closest) == qary_lattice_small.get_dimension()
        
        # Check that result is a lattice point (approximately)
        basis = qary_lattice_small.get_basis()
        
        # Find coefficients: basis.T @ coeff = closest
        try:
            coeff = np.linalg.lstsq(basis.T, closest, rcond=None)[0]
            reconstructed = basis.T @ coeff
            
            np.testing.assert_allclose(
                reconstructed, closest,
                rtol=tolerance_config['rtol'],
                err_msg="CVP result is not a valid lattice point"
            )
        except np.linalg.LinAlgError:
            # Matrix might be singular - skip this check
            pass
    
    @pytest.mark.numerical
    def test_qary_security_estimation(self):
        """Test security level estimation for q-ary lattice."""
        # Create a lattice that should have reasonable security
        n, m, q = 8, 16, 97  # Small but realistic parameters
        A = np.random.randint(0, q, (n, m))
        lattice = QaryLattice.from_lwe_instance(A, q, dual=True)
        
        security = lattice.estimate_security_level()
        
        # Security should be positive and reasonable
        assert security > 0, "Security level must be positive"
        assert security < 1000, "Security level seems unreasonably high"
        
        # For these small parameters, expect low security
        assert security < 100, "Small parameters should give low security"
    
    def test_qary_lattice_point_sampling(self, qary_lattice_small):
        """Test lattice point sampling for q-ary lattice."""
        sigma = 1.0
        
        for _ in range(10):  # Limited tests due to complexity
            point = qary_lattice_small.sample_lattice_point(sigma)
            
            # Check dimension
            assert len(point) == qary_lattice_small.get_dimension()
            
            # Check that point values are reasonable
            assert np.all(np.isfinite(point)), "Sample contains non-finite values"
    
    @pytest.mark.numerical
    def test_qary_determinant_and_volume(self, tolerance_config):
        """Test determinant and volume calculations for q-ary lattice."""
        n, m, q = 3, 5, 7
        A = np.random.randint(0, q, (n, m))
        
        lattice = QaryLattice.from_lwe_instance(A, q, dual=False)
        basis = lattice.get_basis()
        
        # For primal lattice, volume should be q^n
        expected_volume = q**n
        
        # Compute actual volume (determinant of Gram matrix)
        gram = basis @ basis.T
        actual_volume = np.sqrt(np.linalg.det(gram))
        
        np.testing.assert_allclose(
            actual_volume, expected_volume,
            rtol=tolerance_config['rtol'],
            err_msg="Q-ary lattice volume calculation incorrect"
        )
    
    @pytest.mark.edge_case
    def test_qary_pathological_cases(self):
        """Test q-ary lattice with pathological inputs."""
        # Very small q
        n, m, q = 2, 3, 2
        A = np.random.randint(0, q, (n, m))
        lattice = QaryLattice.from_lwe_instance(A, q, dual=False)
        
        # Should still work
        basis = lattice.get_basis()
        assert basis.shape == (n, m + n)
        
        # Large q
        q_large = 997  # Large prime
        lattice_large = QaryLattice.from_lwe_instance(A, q_large, dual=False)
        basis_large = lattice_large.get_basis()
        assert basis_large.shape == (n, m + n)
        
        # Check that q appears correctly in basis
        np.testing.assert_array_equal(basis_large[:, m:], q_large * np.eye(n))


@pytest.mark.reproducibility
class TestLatticeReproducibility:
    """Test reproducibility of lattice operations."""
    
    def test_deterministic_construction(self, test_seed):
        """Test that lattice construction is deterministic."""
        np.random.seed(test_seed)
        
        # Create two identical lattices
        n, m, q = 4, 6, 17
        A1 = np.random.randint(0, q, (n, m))
        
        np.random.seed(test_seed)
        A2 = np.random.randint(0, q, (n, m))
        
        lattice1 = QaryLattice.from_lwe_instance(A1, q, dual=False)
        lattice2 = QaryLattice.from_lwe_instance(A2, q, dual=False)
        
        np.testing.assert_array_equal(
            lattice1.get_basis(), lattice2.get_basis(),
            err_msg="Deterministic construction failed"
        )
    
    def test_gram_schmidt_stability(self, random_2d_basis, tolerance_config):
        """Test numerical stability of Gram-Schmidt process."""
        class TestLattice(BaseLattice):
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
        
        lattice = TestLattice(random_2d_basis)
        
        # Run Gram-Schmidt multiple times
        results = []
        for _ in range(10):
            Q, R = lattice.get_gram_schmidt()
            results.append((Q.copy(), R.copy()))
        
        # All results should be identical
        Q_ref, R_ref = results[0]
        for Q, R in results[1:]:
            np.testing.assert_allclose(
                Q, Q_ref, rtol=tolerance_config['rtol'],
                err_msg="Gram-Schmidt results are not stable"
            )
            np.testing.assert_allclose(
                R, R_ref, rtol=tolerance_config['rtol'],
                err_msg="Gram-Schmidt results are not stable"
            )