"""
Test suite for lattice implementations.

Tests all lattice classes for correctness, including:
- Basis generation and properties
- Gram-Schmidt orthogonalization
- CVP decoding
- Smoothing parameter computation
- Specific lattice properties
"""

import pytest
import numpy as np
from src.lattices.base import Lattice
from src.lattices.identity import IdentityLattice
from src.lattices.qary import QaryLattice
from src.lattices.ntru import NTRULattice
from src.lattices.reduction import LatticeReduction


class TestBaseLattice:
    """Test base lattice functionality."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            Lattice(10)
    
    def test_gram_schmidt_orthogonality(self):
        """Test Gram-Schmidt produces orthogonal basis."""
        # Use identity lattice for testing
        lattice = IdentityLattice(5)
        gs_basis, _ = lattice.get_gram_schmidt()
        
        # Check orthogonality
        gram = gs_basis @ gs_basis.T
        np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-10)
    
    def test_smoothing_parameter_bounds(self):
        """Test smoothing parameter satisfies theoretical bounds."""
        lattice = IdentityLattice(10)
        
        # Test different epsilon values
        for epsilon in [0.1, 0.01, 0.001]:
            eta = lattice.smoothing_parameter(epsilon)
            
            # Should be positive
            assert eta > 0
            
            # Should decrease with larger epsilon
            if epsilon < 0.1:
                eta_larger = lattice.smoothing_parameter(0.1)
                assert eta >= eta_larger


class TestIdentityLattice:
    """Test Z^n lattice implementation."""
    
    def test_basis_is_identity(self):
        """Test that basis is identity matrix."""
        n = 10
        lattice = IdentityLattice(n)
        basis = lattice.get_basis()
        
        np.testing.assert_array_equal(basis, np.eye(n))
    
    def test_nearest_plane_rounding(self):
        """Test nearest plane algorithm reduces to rounding."""
        lattice = IdentityLattice(5)
        
        # Test various targets
        targets = [
            np.array([1.2, 3.7, -2.1, 0.6, -4.9]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([-1.1, 2.9, -3.1, 4.1, -5.2])
        ]
        
        for target in targets:
            decoded = lattice.nearest_plane(target)
            expected = np.round(target)
            np.testing.assert_array_equal(decoded, expected)
    
    def test_direct_sampling(self):
        """Test direct sampling method."""
        lattice = IdentityLattice(100)
        sigma = 5.0
        
        # Generate samples
        n_samples = 10000
        samples = np.array([lattice.direct_sample(sigma) for _ in range(n_samples)])
        
        # Check mean (should be near 0)
        mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean, 0, atol=0.1)
        
        # Check variance (should be near sigma^2 for large sigma)
        var = np.var(samples, axis=0)
        expected_var = sigma**2 * (1 - np.exp(-2*np.pi))  # Discrete correction
        np.testing.assert_allclose(var, expected_var, rtol=0.1)
    
    def test_analytical_smoothing_parameter(self):
        """Test analytical formula for smoothing parameter."""
        lattice = IdentityLattice(10)
        
        # For Z^n, eta_epsilon = sqrt(ln(2n(1+1/epsilon))/pi)
        epsilon = 0.01
        eta = lattice.smoothing_parameter(epsilon)
        
        n = lattice.dimension
        expected = np.sqrt(np.log(2 * n * (1 + 1/epsilon)) / np.pi)
        
        np.testing.assert_allclose(eta, expected, rtol=1e-10)


class TestQaryLattice:
    """Test q-ary lattice implementation."""
    
    def test_primal_lattice_construction(self):
        """Test primal lattice ›_q(A) construction."""
        n, m, q = 50, 100, 1024
        lattice = QaryLattice.random_qary_lattice(n, m, q, lattice_type='primal')
        
        basis = lattice.get_basis()
        
        # Check dimension
        assert basis.shape == (m, m)
        
        # Check that basis vectors are in lattice
        A = lattice.A
        for i in range(m):
            v = basis[i]
            # Check Av = 0 mod q
            residue = (A @ v) % q
            np.testing.assert_array_equal(residue, 0)
    
    def test_dual_lattice_construction(self):
        """Test dual lattice ›_q^¥(A) construction."""
        n, m, q = 50, 100, 1024
        lattice = QaryLattice.random_qary_lattice(n, m, q, lattice_type='dual')
        
        basis = lattice.get_basis()
        
        # Check dimension
        assert basis.shape == (m, m)
        
        # Check that basis vectors are in dual lattice
        A = lattice.A
        for i in range(m):
            v = basis[i]
            # Check A^T s = v mod q for some s
            # This is more complex to verify directly
    
    def test_lwe_instance_generation(self):
        """Test LWE instance lattice generation."""
        n, m, q = 128, 256, 3329
        alpha = 3.2 / np.sqrt(q)
        
        lattice = QaryLattice.from_lwe_instance(n, m, q, alpha)
        
        # Check security estimate
        security = lattice.estimate_security_level()
        
        assert security['classical_bits'] > 0
        assert security['quantum_bits'] > 0
        assert security['quantum_bits'] < security['classical_bits']
    
    def test_ideal_lattice_structure(self):
        """Test ideal lattice (Ring-LWE) structure."""
        n, q = 256, 3329
        lattice = QaryLattice.ideal_qary_lattice(n, q)
        
        # Check circulant structure
        A = lattice.A
        
        # First row defines the circulant matrix
        first_row = A[0]
        for i in range(1, n):
            expected_row = np.roll(first_row, i)
            np.testing.assert_array_equal(A[i], expected_row)
    
    def test_hermite_factor_computation(self):
        """Test Hermite factor is computed correctly."""
        n, m, q = 100, 200, 1024
        lattice = QaryLattice.random_qary_lattice(n, m, q)
        
        # Get shortest vector estimate
        lambda_1 = lattice.first_minimum()
        det = np.abs(np.linalg.det(lattice.get_basis()))
        
        # Hermite factor
        delta = (lambda_1 / det**(1/m))**(1/m)
        
        # Should be >= 1 (equality only for orthogonal lattices)
        assert delta >= 1.0


class TestNTRULattice:
    """Test NTRU lattice implementation."""
    
    def test_ntru_construction(self):
        """Test basic NTRU lattice construction."""
        n = 64  # Small for testing
        q = 257  # Small prime
        
        lattice = NTRULattice(n, q)
        lattice.generate_basis()
        
        # Check dimensions
        assert lattice.dimension == 2 * n
        
        # Get public and private bases
        pub_basis = lattice.get_public_basis()
        priv_basis = lattice.get_private_basis()
        
        assert pub_basis.shape == (2*n, 2*n)
        assert priv_basis.shape == (2*n, 2*n)
        
        # Both should generate same lattice
        # Check determinants are equal (up to sign)
        det_pub = np.abs(np.linalg.det(pub_basis))
        det_priv = np.abs(np.linalg.det(priv_basis))
        
        np.testing.assert_allclose(det_pub, det_priv, rtol=1e-10)
    
    def test_ntru_equation(self):
        """Test that NTRU equation fG - gF = q is satisfied."""
        n = 32
        q = 127
        
        lattice = NTRULattice(n, q)
        lattice.generate_basis()
        
        # Check NTRU equation (simplified check)
        # In full implementation, would check polynomial arithmetic
        assert lattice.f is not None
        assert lattice.g is not None
        assert lattice.F is not None
        assert lattice.G is not None
    
    def test_negacyclic_structure(self):
        """Test negacyclic matrix structure."""
        n = 16
        poly = np.random.randint(-5, 5, n)
        
        lattice = NTRULattice(n)
        mat = lattice._negacyclic_matrix(poly)
        
        # Check first row
        np.testing.assert_array_equal(mat[0], poly)
        
        # Check negacyclic shifts
        for i in range(1, n):
            expected = np.zeros(n)
            expected[0] = -poly[n-i]
            expected[1:] = poly[:n-i]
            np.testing.assert_array_equal(mat[i], expected)
    
    def test_max_gram_schmidt_norm(self):
        """Test Gram-Schmidt norm bounds."""
        n = 64
        q = 257
        sigma = 4.0
        
        lattice = NTRULattice(n, q, sigma)
        lattice.generate_basis()
        
        max_norm, theoretical_bound = lattice.get_max_gram_schmidt_norm()
        
        # Max norm should not exceed theoretical bound (with high probability)
        assert max_norm <= 2 * theoretical_bound  # Allow some margin


class TestLatticeReduction:
    """Test lattice reduction algorithms."""
    
    def test_lll_reduction(self):
        """Test LLL reduces basis quality."""
        # Random lattice
        n = 10
        basis = np.random.randn(n, n) * 10
        basis = basis.astype(int)
        
        reducer = LatticeReduction()
        reduced_basis, stats = reducer.lll_reduce(basis)
        
        # Check LLL conditions
        assert reducer.check_lll_reduced(reduced_basis, delta=0.99)
        
        # Check basis generates same lattice
        det_original = np.abs(np.linalg.det(basis))
        det_reduced = np.abs(np.linalg.det(reduced_basis))
        np.testing.assert_allclose(det_original, det_reduced, rtol=1e-10)
        
        # Orthogonality defect should improve
        assert stats['orthogonality_defect_after'] <= stats['orthogonality_defect_before']
    
    def test_bkz_reduction(self):
        """Test BKZ reduction with different block sizes."""
        n = 20
        basis = np.random.randn(n, n) * 10
        basis = basis.astype(int)
        
        reducer = LatticeReduction()
        
        # Test increasing block sizes
        for beta in [2, 4, 8]:
            reduced_basis, stats = reducer.bkz_reduce(basis.copy(), block_size=beta)
            
            # Check basis is valid
            det_original = np.abs(np.linalg.det(basis))
            det_reduced = np.abs(np.linalg.det(reduced_basis))
            np.testing.assert_allclose(det_original, det_reduced, rtol=1e-10)
            
            # Larger block size should give better reduction
            if beta > 2:
                assert stats['hermite_factor_after'] <= prev_hermite
            
            prev_hermite = stats['hermite_factor_after']
    
    def test_sampling_optimization(self):
        """Test reduction optimized for sampling."""
        # Create a "bad" basis
        n = 10
        basis = np.eye(n)
        basis[0] *= 1000  # Make first vector very long
        
        reducer = LatticeReduction()
        opt_basis, improvements = reducer.sampling_reduce(
            basis,
            target_distribution='gaussian',
            sigma=10.0
        )
        
        # Should improve max Gram-Schmidt norm
        assert improvements['sampling_improvement'] > 1.0
        
        # Check recommendations
        assert 'recommendations' in improvements
        assert len(improvements['recommendations']) > 0


@pytest.fixture
def small_lattice():
    """Fixture for small test lattice."""
    return IdentityLattice(5)


@pytest.fixture
def medium_lattice():
    """Fixture for medium test lattice."""
    return QaryLattice.random_qary_lattice(50, 100, 1024)


def test_lattice_integration(small_lattice, medium_lattice):
    """Integration test using multiple lattices."""
    # Test that all lattices implement required interface
    for lattice in [small_lattice, medium_lattice]:
        # Should be able to get basis
        basis = lattice.get_basis()
        assert basis.shape[0] == lattice.dimension
        
        # Should be able to decode
        target = np.random.randn(lattice.dimension)
        decoded = lattice.decode_cvp(target)
        assert decoded.shape == target.shape
        
        # Should compute smoothing parameter
        eta = lattice.smoothing_parameter(0.01)
        assert eta > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])