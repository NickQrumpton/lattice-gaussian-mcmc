#!/usr/bin/env python3
"""
NTRU Lattice Implementation for Research.

This module provides a complete NTRU lattice implementation suitable for
cryptographic research and lattice Gaussian sampling experiments.

The implementation includes:
- Polynomial arithmetic in Z_q[x]/(x^n + 1)
- NTRU key generation with ternary secrets
- Lattice basis construction
- Gram-Schmidt orthogonalization
- Integration with lattice Gaussian samplers

Note: This implementation requires SageMath to be run as a .sage file.
"""

from typing import Optional, Tuple, List
import numpy as np


class NTRULattice:
    """
    NTRU lattice for cryptographic applications.
    
    Implements the NTRU lattice construction used in the FALCON signature
    scheme and suitable for lattice Gaussian sampling research.
    
    Attributes:
        n (int): Ring dimension (power of 2)
        q (int): Modulus (prime)
        sigma (float): Gaussian parameter for key generation
        f, g: Secret key polynomials
        h: Public key polynomial (g/f mod q)
        basis_matrix: 2n × 2n lattice basis
    """
    
    def __init__(self, n: int, q: int, sigma: Optional[float] = None):
        """
        Initialize NTRU lattice.
        
        Args:
            n: Ring dimension (should be power of 2)
            q: Modulus (should be prime)
            sigma: Gaussian parameter (default: 1.17 * sqrt(q))
        """
        self.n = n
        self.q = q
        self.sigma = sigma if sigma is not None else 1.17 * np.sqrt(q)
        
        # These will be set by SageMath when loaded as .sage
        self.R = None  # Z[x]/(x^n + 1)
        self.Rq = None  # Z_q[x]/(x^n + 1)
        self.dgauss = None  # Discrete Gaussian sampler
        
        # Keys
        self.f = None
        self.g = None
        self.h = None
        self.basis_matrix = None
        
        self._initialized = False
    
    def _initialize_sage(self):
        """Initialize SageMath-specific components."""
        if self._initialized:
            return
            
        from sage.all import PolynomialRing, GF, ZZ
        from sage.stats.distributions.discrete_gaussian_integer import (
            DiscreteGaussianDistributionIntegerSampler
        )
        
        # Create polynomial rings
        R = PolynomialRing(ZZ, 'x')
        x = R.gen()
        self.R = R.quotient(x**self.n + 1)
        
        Rq = PolynomialRing(GF(self.q), 'x')
        x = Rq.gen()
        self.Rq = Rq.quotient(x**self.n + 1)
        
        # Discrete Gaussian sampler
        self.dgauss = DiscreteGaussianDistributionIntegerSampler(self.sigma)
        
        self._initialized = True
    
    def generate_keys(self, key_type: str = 'ternary') -> bool:
        """
        Generate NTRU key pair.
        
        Args:
            key_type: Type of keys ('ternary' or 'gaussian')
            
        Returns:
            True if successful, False otherwise
        """
        self._initialize_sage()
        
        max_attempts = 100
        for _ in range(max_attempts):
            if key_type == 'ternary':
                # Sample ternary f, g
                f_coeffs = self._sample_ternary(ensure_invertible=True)
                g_coeffs = self._sample_ternary()
            else:
                # Sample Gaussian f, g
                f_coeffs = self._sample_gaussian(ensure_invertible=True)
                g_coeffs = self._sample_gaussian()
            
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
            from sage.all import ZZ
            h_coeffs = [ZZ(c) for c in list(h_q.lift())]
            self.h = self.R(h_coeffs)
            
            # Generate basis matrix
            self._generate_basis()
            return True
        
        return False
    
    def _sample_ternary(self, ensure_invertible: bool = False) -> List[int]:
        """Sample polynomial with ternary coefficients {-1, 0, 1}."""
        from sage.all import choice
        
        coeffs = [choice([-1, 0, 1]) for _ in range(self.n)]
        
        if ensure_invertible:
            # Set f[0] = 1 to increase chance of invertibility
            coeffs[0] = 1
            
        return coeffs
    
    def _sample_gaussian(self, ensure_invertible: bool = False) -> List[int]:
        """Sample polynomial with discrete Gaussian coefficients."""
        coeffs = [self.dgauss() for _ in range(self.n)]
        
        if ensure_invertible:
            # Add 1 to f[0] to increase chance of invertibility
            coeffs[0] += 1
            
        return coeffs
    
    def _generate_basis(self):
        """
        Generate NTRU lattice basis.
        
        For simplicity, we use the standard construction:
        B = [[q*I, 0], [H, I]]
        
        where H is the circulant matrix of h.
        """
        from sage.all import identity_matrix, zero_matrix, matrix, block_matrix, ZZ
        
        # Identity and zero matrices
        I_n = identity_matrix(ZZ, self.n)
        Z_n = zero_matrix(ZZ, self.n)
        
        # Create circulant matrix from h
        h_coeffs = list(self.h.lift())[:self.n]
        H = matrix(ZZ, self.n, self.n)
        
        for i in range(self.n):
            for j in range(self.n):
                idx = (j - i) % self.n
                H[i, j] = h_coeffs[idx] if idx < len(h_coeffs) else 0
        
        # Construct basis [[q*I, 0], [H, I]]
        self.basis_matrix = block_matrix([
            [self.q * I_n, Z_n],
            [H, I_n]
        ])
    
    def get_basis(self) -> 'Matrix':
        """Return the lattice basis matrix."""
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        return self.basis_matrix
    
    def gram_schmidt_norms(self) -> List[float]:
        """
        Compute Gram-Schmidt norms of basis vectors.
        
        Returns:
            List of Gram-Schmidt norms
        """
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        
        from sage.all import RDF, vector
        
        B = self.basis_matrix.change_ring(RDF)
        gs_norms = []
        gs_vecs = []
        
        for i in range(B.nrows()):
            v = vector(B[i])
            
            # Orthogonalize against previous vectors
            for j in range(len(gs_vecs)):
                if gs_norms[j] > 0:
                    proj_coeff = v.dot_product(gs_vecs[j]) / (gs_norms[j]**2)
                    v = v - proj_coeff * gs_vecs[j]
            
            gs_vecs.append(v)
            gs_norms.append(float(v.norm()))
        
        return gs_norms
    
    def babai_nearest_plane(self, target: 'vector') -> 'vector':
        """
        Find closest lattice point using Babai's nearest plane algorithm.
        
        Args:
            target: Target vector
            
        Returns:
            Closest lattice vector
        """
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        
        from sage.all import vector, round
        
        # Solve for coefficients
        coeffs = self.basis_matrix.solve_left(target)
        
        # Round to nearest integers
        rounded_coeffs = vector([round(c) for c in coeffs])
        
        # Compute lattice point
        return self.basis_matrix.transpose() * rounded_coeffs
    
    def sample_discrete_gaussian(self, sigma: float, num_samples: int = 1) -> List['vector']:
        """
        Sample from discrete Gaussian distribution over the lattice.
        
        Args:
            sigma: Gaussian parameter
            num_samples: Number of samples
            
        Returns:
            List of lattice vectors
        """
        if self.basis_matrix is None:
            raise ValueError("Keys not generated yet")
        
        from sage.all import vector, normalvariate
        
        samples = []
        
        for _ in range(num_samples):
            # Sample continuous Gaussian
            c = vector([sigma * normalvariate(0, 1) for _ in range(2 * self.n)])
            
            # Find closest lattice point
            v = self.babai_nearest_plane(c)
            samples.append(v)
        
        return samples


# Example usage (when loaded as .sage file)
if __name__ == "__main__":
    print("NTRU Lattice Example")
    print("=" * 50)
    
    # Small example
    ntru = NTRULattice(n=64, q=12289)
    print(f"Created NTRU lattice: n={ntru.n}, q={ntru.q}")
    
    print("\nGenerating keys...")
    if ntru.generate_keys(key_type='ternary'):
        print("✓ Keys generated successfully")
        
        # Get basis
        B = ntru.get_basis()
        print(f"\nBasis dimensions: {B.nrows()} × {B.ncols()}")
        
        # Compute GS norms
        gs_norms = ntru.gram_schmidt_norms()
        print(f"\nGram-Schmidt norms:")
        print(f"  Min: {min(gs_norms):.2f}")
        print(f"  Max: {max(gs_norms):.2f}")
        print(f"  Ratio: {max(gs_norms)/min(gs_norms):.2f}")
        
        # Sample from discrete Gaussian
        print("\nSampling from discrete Gaussian...")
        samples = ntru.sample_discrete_gaussian(sigma=100, num_samples=10)
        norms = [float(s.norm()) for s in samples]
        print(f"  Average norm: {np.mean(norms):.2f}")
        print(f"  Std dev: {np.std(norms):.2f}")
    else:
        print("✗ Key generation failed")