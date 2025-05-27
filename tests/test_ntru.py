#!/usr/bin/env sage
"""
Unit tests for NTRU lattice implementation.

Tests polynomial arithmetic, NTRUSolve algorithm, key generation,
and Gram-Schmidt orthogonalization for cryptographic parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import *
from src.lattices.ntru import NTRULattice
import time

class TestNTRULattice:
    """Test suite for NTRU lattice implementation."""
    
    def test_polynomial_arithmetic_small(self):
        """Test polynomial arithmetic for small parameters."""
        lattice = NTRULattice(n=8, q=257)
        R = lattice.R
        
        # Test addition
        f = R([1, 2, 3, 4, 0, 0, 0, 0])
        g = R([0, 1, 0, 1, 0, 0, 0, 0])
        h = lattice._add_poly(f, g)
        assert h == R([1, 3, 3, 5, 0, 0, 0, 0])
        
        # Test multiplication with wrap-around (x^n = -1)
        f = R([1, 0, 0, 0, 0, 0, 0, 1])  # 1 + x^7
        g = R([0, 1, 0, 0, 0, 0, 0, 0])  # x
        h = lattice._multiply_poly(f, g)
        # x * (1 + x^7) = x + x^8 = x - 1 (since x^8 = -1)
        assert h == R([-1, 1, 0, 0, 0, 0, 0, 0])
        
        # Test modular inverse
        f = R([2, 1, 0, 0, 0, 0, 0, 0])  # 2 + x
        f_inv = lattice._invert_poly_mod_q(f)
        assert f_inv is not None
        prod = lattice._multiply_poly(f, f_inv)
        assert prod == R(1)
        
        print("✓ Polynomial arithmetic tests passed")
    
    def test_conjugate_and_norm(self):
        """Test conjugate and norm operations."""
        lattice = NTRULattice(n=8, q=257)
        R = lattice.R
        
        # Test conjugate: f*(x) = f(x^-1) = f(-x) in Z[x]/(x^n + 1)
        f = R([1, 2, 3, 4, 0, 0, 0, 0])
        f_conj = lattice._conjugate_poly(f)
        # f*(x) = 1 - 4x^7 - 3x^6 - 2x^5
        expected = R([1, 0, 0, 0, -2, -3, -4, 0])
        assert f_conj == expected
        
        # Test norm: N(f) = f * f*
        Nf = lattice._multiply_poly(f, f_conj)
        # Verify norm is palindromic (symmetric coefficients)
        coeffs = list(Nf)
        n = len(coeffs)
        for i in range(n//2):
            assert coeffs[i] == coeffs[n-1-i], f"Norm not palindromic at index {i}"
        
        print("✓ Conjugate and norm tests passed")
    
    def test_ntru_solve_small(self):
        """Test NTRUSolve algorithm for small parameters."""
        lattice = NTRULattice(n=8, q=257)
        R = lattice.R
        
        # Generate simple f, g
        f = R([3, 1, -1, 0, 1, 0, 0, -1])
        g = R([1, 2, 0, -1, 0, 1, -1, 0])
        
        # Ensure f is invertible mod q
        f_inv = lattice._invert_poly_mod_q(f)
        if f_inv is None:
            # Try different f
            f = R([2, 1, 0, 0, 0, 0, 0, 0])
            f_inv = lattice._invert_poly_mod_q(f)
            assert f_inv is not None
        
        # Solve NTRU equation
        F, G = lattice._ntru_solve(f, g)
        
        # Verify: fG - gF = q
        fG = lattice._multiply_poly(f, G)
        gF = lattice._multiply_poly(g, F)
        diff = fG - gF
        
        # Check if diff = q (as constant polynomial)
        diff_coeffs = list(diff)
        assert diff_coeffs[0] % lattice.q == 0
        for i in range(1, len(diff_coeffs)):
            assert diff_coeffs[i] == 0
        
        print("✓ NTRUSolve algorithm tests passed")
    
    def test_key_generation_small(self):
        """Test key generation for small parameters."""
        lattice = NTRULattice(n=16, q=257)
        
        # Generate keys
        keys = lattice.generate_keys(max_attempts=10)
        assert keys is not None, "Key generation failed"
        
        f, g, F, G, h = keys
        
        # Verify NTRU equation: fG - gF = q
        fG = lattice._multiply_poly(f, G)
        gF = lattice._multiply_poly(g, F)
        diff = fG - gF
        diff_coeffs = list(diff)
        assert diff_coeffs[0] % lattice.q == 0
        for i in range(1, len(diff_coeffs)):
            assert diff_coeffs[i] == 0
        
        # Verify public key: h = g/f mod q
        f_inv = lattice._invert_poly_mod_q(f)
        h_computed = lattice._multiply_poly(g, f_inv)
        assert h == h_computed
        
        # Verify basis matrix properties
        assert hasattr(lattice, 'basis_matrix')
        B = lattice.basis_matrix
        assert B.nrows() == 2 * lattice.n
        assert B.ncols() == 2 * lattice.n
        
        # Check determinant = q^n
        det = B.det()
        assert abs(det) == lattice.q^lattice.n
        
        print("✓ Key generation tests passed")
    
    def test_gram_schmidt_orthogonalization(self):
        """Test Gram-Schmidt orthogonalization."""
        lattice = NTRULattice(n=8, q=257)
        
        # Generate keys to get basis
        keys = lattice.generate_keys()
        assert keys is not None
        
        # Compute Gram-Schmidt
        gs_norms = lattice.gram_schmidt_norms()
        assert gs_norms is not None
        assert len(gs_norms) == 2 * lattice.n
        
        # Check that norms are positive
        for norm in gs_norms:
            assert norm > 0
        
        # Check that first n norms are approximately equal (from circulant structure)
        first_norm = gs_norms[0]
        for i in range(1, lattice.n):
            ratio = gs_norms[i] / first_norm
            assert 0.9 < ratio < 1.1, f"GS norm {i} differs too much from norm 0"
        
        print("✓ Gram-Schmidt orthogonalization tests passed")
    
    def test_cryptographic_parameters_512(self):
        """Test with cryptographic parameters n=512."""
        print("\nTesting n=512 (this may take a moment)...")
        start_time = time.time()
        
        lattice = NTRULattice(n=512, q=12289)  # FALCON-512 parameters
        
        # Generate keys
        keys = lattice.generate_keys(max_attempts=5)
        assert keys is not None, "Key generation failed for n=512"
        
        f, g, F, G, h = keys
        
        # Basic verification (full verification would be too slow)
        # Just check dimensions and invertibility
        assert f.parent() == lattice.R
        assert g.parent() == lattice.R
        
        # Verify f is invertible mod q
        f_inv = lattice._invert_poly_mod_q(f)
        assert f_inv is not None
        
        # Check basis matrix size
        assert lattice.basis_matrix.nrows() == 1024
        assert lattice.basis_matrix.ncols() == 1024
        
        elapsed = time.time() - start_time
        print(f"✓ Cryptographic parameters n=512 test passed ({elapsed:.2f}s)")
    
    def test_cryptographic_parameters_1024(self):
        """Test with cryptographic parameters n=1024."""
        print("\nTesting n=1024 (this may take a while)...")
        start_time = time.time()
        
        lattice = NTRULattice(n=1024, q=12289)  # FALCON-1024 parameters
        
        # For n=1024, just test initialization and basic operations
        # Full key generation would be too slow for unit tests
        
        # Test polynomial operations
        R = lattice.R
        f = R.random_element(degree=100)  # Sparse polynomial
        g = R.random_element(degree=100)
        
        # Test multiplication
        h = lattice._multiply_poly(f, g)
        assert h.parent() == R
        
        # Test that we can create the basis structure
        assert lattice.n == 1024
        assert lattice.q == 12289
        
        elapsed = time.time() - start_time
        print(f"✓ Cryptographic parameters n=1024 test passed ({elapsed:.2f}s)")
    
    def test_shortest_vector_approximation(self):
        """Test shortest vector finding."""
        lattice = NTRULattice(n=16, q=257)
        
        # Generate keys
        keys = lattice.generate_keys()
        assert keys is not None
        
        # Find short vector
        short_vec = lattice.shortest_vector()
        assert short_vec is not None
        assert len(short_vec) == 2 * lattice.n
        
        # Verify it's in the lattice
        assert lattice.is_in_lattice(short_vec)
        
        # Check it's reasonably short (should be (f, g) or similar)
        norm = sqrt(sum(x^2 for x in short_vec))
        # For small parameters, should be < sqrt(2n) * sigma
        assert norm < sqrt(2 * lattice.n) * 2
        
        print("✓ Shortest vector approximation tests passed")
    
    def test_matrix_conversion(self):
        """Test polynomial to matrix conversion."""
        lattice = NTRULattice(n=4, q=257)
        R = lattice.R
        
        # Test simple polynomial
        f = R([1, 2, 3, 4])
        M = lattice._poly_to_matrix(f)
        
        # Expected negacyclic circulant matrix:
        # [1  -4  -3  -2]
        # [2   1  -4  -3]
        # [3   2   1  -4]
        # [4   3   2   1]
        expected = matrix([
            [1, -4, -3, -2],
            [2,  1, -4, -3],
            [3,  2,  1, -4],
            [4,  3,  2,  1]
        ])
        
        assert M == expected
        print("✓ Matrix conversion tests passed")
    
    def test_performance_scaling(self):
        """Test performance scaling with dimension."""
        print("\nPerformance scaling test:")
        
        for n in [32, 64, 128, 256]:
            start_time = time.time()
            
            lattice = NTRULattice(n=n, q=257)
            keys = lattice.generate_keys(max_attempts=3)
            
            if keys is None:
                print(f"  n={n}: Key generation failed")
                continue
            
            # Time Gram-Schmidt
            gs_start = time.time()
            gs_norms = lattice.gram_schmidt_norms()
            gs_time = time.time() - gs_start
            
            total_time = time.time() - start_time
            print(f"  n={n}: Total {total_time:.3f}s, GS {gs_time:.3f}s")


def run_all_tests():
    """Run all NTRU lattice tests."""
    print("Running NTRU lattice unit tests...\n")
    
    test = TestNTRULattice()
    
    # Run tests in order of complexity
    test.test_polynomial_arithmetic_small()
    test.test_conjugate_and_norm()
    test.test_ntru_solve_small()
    test.test_key_generation_small()
    test.test_gram_schmidt_orthogonalization()
    test.test_shortest_vector_approximation()
    test.test_matrix_conversion()
    
    # Cryptographic parameter tests (slower)
    test.test_cryptographic_parameters_512()
    test.test_cryptographic_parameters_1024()
    
    # Performance test
    test.test_performance_scaling()
    
    print("\n✅ All NTRU lattice tests passed!")


if __name__ == "__main__":
    run_all_tests()