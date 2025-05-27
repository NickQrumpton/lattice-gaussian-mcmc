#!/usr/bin/env python3
"""
NTRU Lattice Implementation for SageMath.

This module provides a complete NTRU lattice implementation suitable for
research in lattice-based cryptography and Gaussian sampling.

To use this module:
    sage: load('ntru_clean.py')
    sage: ntru = NTRULattice(n=64, q=12289)
    sage: ntru.generate_keys()
    sage: basis = ntru.get_basis()
"""

# This file must be run with SageMath
try:
    from sage.all import *
except ImportError:
    raise ImportError("This module requires SageMath")

from typing import Optional, Tuple, List
import warnings


class NTRULattice:
    """
    NTRU lattice for cryptographic applications.
    
    Implements the standard NTRU lattice construction with basis:
        B = [[q*I, 0], [H, I]]
    
    where H is derived from the public key h = g/f mod q.
    """
    
    def __init__(self, n: int, q: int, sigma: Optional[float] = None):
        """
        Initialize NTRU lattice.
        
        Args:
            n: Ring dimension (power of 2)
            q: Modulus (prime)
            sigma: Gaussian parameter for key generation
        """
        self.n = n
        self.q = q
        self.sigma = sigma if sigma is not None else 1.17 * sqrt(q).n()
        
        # Polynomial rings
        R = PolynomialRing(ZZ, 'x')
        x = R.gen()
        self.R = R.quotient(x**n + 1)
        
        Rq = PolynomialRing(GF(q), 'x')
        x = Rq.gen()
        self.Rq = Rq.quotient(x**n + 1)
        
        # Discrete Gaussian sampler
        from sage.stats.distributions.discrete_gaussian_integer import (
            DiscreteGaussianDistributionIntegerSampler
        )
        self.dgauss = DiscreteGaussianDistributionIntegerSampler(self.sigma)
        
        # Keys (to be generated)
        self.f = None
        self.g = None
        self.h = None
        self.basis_matrix = None
    
    def generate_keys(self, key_type: str = 'ternary', max_attempts: int = 100) -> bool:
        """
        Generate NTRU keys.
        
        Args:
            key_type: 'ternary' for {-1,0,1} coefficients, 'gaussian' for Gaussian
            max_attempts: Maximum attempts to find invertible f
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_attempts):
            # Sample f and g
            if key_type == 'ternary':
                f_coeffs = [choice([-1, 0, 1]) for _ in range(self.n)]
                f_coeffs[0] = 1  # Make f[0] = 1 for better invertibility
                g_coeffs = [choice([-1, 0, 1]) for _ in range(self.n)]
            else:
                f_coeffs = [self.dgauss() for _ in range(self.n)]
                f_coeffs[0] += 1  # Bias towards invertibility
                g_coeffs = [self.dgauss() for _ in range(self.n)]
            
            self.f = self.R(f_coeffs)
            self.g = self.R(g_coeffs)
            
            # Check if f is invertible mod q
            f_q = self.Rq(f_coeffs)
            try:
                f_inv_q = f_q**(-1)
            except:
                continue
            
            # Compute public key h = g/f mod q
            g_q = self.Rq(g_coeffs)
            h_q = g_q * f_inv_q
            
            # Convert back to integer coefficients
            h_coeffs = [ZZ(c) for c in list(h_q.lift())]
            self.h = self.R(h_coeffs)
            
            # Generate basis
            self._generate_basis()
            return True
        
        warnings.warn(f"Failed to generate keys after {max_attempts} attempts")
        return False
    
    def _generate_basis(self):
        """Generate the NTRU lattice basis matrix."""
        # Create identity and zero matrices
        I_n = identity_matrix(ZZ, self.n)
        Z_n = zero_matrix(ZZ, self.n)
        
        # Create circulant matrix H from public key h
        h_coeffs = self._get_coeffs(self.h)
        H = matrix(ZZ, self.n, self.n)
        
        for i in range(self.n):
            for j in range(self.n):
                H[i, j] = h_coeffs[(j - i) % self.n]
        
        # Build basis [[q*I, 0], [H, I]]
        self.basis_matrix = block_matrix([
            [self.q * I_n, Z_n],
            [H, I_n]
        ])
    
    def _get_coeffs(self, poly, length: Optional[int] = None) -> List[int]:
        """Extract coefficients from polynomial, padding if needed."""
        if length is None:
            length = self.n
        
        coeffs = list(poly.lift())
        while len(coeffs) < length:
            coeffs.append(0)
        return coeffs[:length]
    
    def get_basis(self) -> 'Matrix':
        """Get the lattice basis matrix."""
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        return self.basis_matrix
    
    def gram_schmidt_norms(self) -> List[float]:
        """Compute Gram-Schmidt norms of basis vectors."""
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        
        B = self.basis_matrix.change_ring(RDF)
        gs_norms = []
        gs_vecs = []
        
        for i in range(B.nrows()):
            v = vector(B[i])
            
            # Orthogonalize
            for j in range(len(gs_vecs)):
                if gs_norms[j] > 0:
                    proj = (v.dot_product(gs_vecs[j]) / gs_norms[j]**2) * gs_vecs[j]
                    v = v - proj
            
            gs_vecs.append(v)
            gs_norms.append(float(v.norm()))
        
        return gs_norms
    
    def closest_vector(self, target: 'vector') -> 'vector':
        """Find closest lattice vector using Babai's algorithm."""
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        
        # Solve for coefficients
        coeffs = self.basis_matrix.solve_left(target)
        
        # Round to nearest integers
        rounded = vector([round(c) for c in coeffs])
        
        # Return lattice vector
        return self.basis_matrix.transpose() * rounded
    
    def sample_discrete_gaussian(self, sigma: float, num_samples: int = 1):
        """Sample from discrete Gaussian over the lattice."""
        samples = []
        
        for _ in range(num_samples):
            # Sample continuous Gaussian
            c = vector([normalvariate(0, sigma) for _ in range(2 * self.n)])
            
            # Find closest lattice point
            v = self.closest_vector(c)
            samples.append(v)
        
        return samples if num_samples > 1 else samples[0]
    
    def verify_basis(self) -> bool:
        """Verify the basis is valid."""
        if self.basis_matrix is None:
            return False
        
        # Check dimensions
        if self.basis_matrix.nrows() != 2 * self.n:
            return False
        if self.basis_matrix.ncols() != 2 * self.n:
            return False
        
        # Check determinant (should be q^n)
        det = abs(self.basis_matrix.det())
        expected = self.q**self.n
        
        if det != expected:
            warnings.warn(f"Determinant {det} != expected {expected}")
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    print("NTRU Lattice Implementation Test")
    print("=" * 50)
    
    # Test with small parameters
    n, q = 64, 12289
    print(f"\nTesting with n={n}, q={q}")
    
    ntru = NTRULattice(n=n, q=q)
    
    print("Generating keys...")
    if ntru.generate_keys(key_type='ternary'):
        print("✓ Keys generated successfully")
        
        # Verify basis
        if ntru.verify_basis():
            print("✓ Basis verified")
        
        # Get basis info
        B = ntru.get_basis()
        print(f"\nBasis dimensions: {B.nrows()} × {B.ncols()}")
        
        # Compute GS norms
        print("\nComputing Gram-Schmidt norms...")
        gs_norms = ntru.gram_schmidt_norms()
        print(f"Min GS norm: {min(gs_norms):.2f}")
        print(f"Max GS norm: {max(gs_norms):.2f}")
        print(f"GS ratio: {max(gs_norms)/min(gs_norms):.2f}")
        
        # Test sampling
        print("\nTesting discrete Gaussian sampling...")
        sample = ntru.sample_discrete_gaussian(sigma=100)
        print(f"Sample norm: {float(sample.norm()):.2f}")
        
        # Verify sample is in lattice
        try:
            coeffs = B.solve_left(sample)
            is_integral = all(abs(c - round(c)) < 1e-10 for c in coeffs)
            print(f"Sample in lattice: {is_integral}")
        except:
            print("Sample verification failed")
    else:
        print("✗ Key generation failed")