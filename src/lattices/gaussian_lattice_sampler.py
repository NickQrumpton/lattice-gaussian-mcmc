#!/usr/bin/env python3
"""
Integration layer for discrete Gaussian sampling on lattices.

This module provides a unified interface for sampling from discrete Gaussian
distributions over various lattice types (Identity, q-ary, NTRU) using the
exact samplers from discrete_gaussian.py.
"""

try:
    from sage.all import *
except ImportError:
    raise ImportError("This module requires SageMath")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.discrete_gaussian import (
    RejectionSampler, CDTSampler, DiscreteGaussianVectorSampler,
    sample_discrete_gaussian_vec
)
from typing import Optional, Union, List


class IdentityLatticeSampler:
    """
    Discrete Gaussian sampler for the identity lattice Z^n.
    
    This is the simplest case where we sample independently from
    each coordinate.
    """
    
    def __init__(self, n: int, sigma: Union[float, List[float]], 
                 center: Optional['vector'] = None):
        """
        Initialize sampler for identity lattice.
        
        Args:
            n: Dimension
            sigma: Standard deviation (scalar or per-coordinate)
            center: Center vector (default: zero)
            
        EXAMPLES::
        
            sage: sampler = IdentityLatticeSampler(n=10, sigma=2.0)
            sage: v = sampler.sample()
            sage: v.parent()
            Ambient free module of rank 10 over Integer Ring
        """
        self.n = n
        self.sigma = sigma
        self.center = center if center is not None else vector(ZZ, n)
        
        # Create vector sampler
        self.vec_sampler = DiscreteGaussianVectorSampler(
            sigma=sigma, center=self.center, n=n
        )
    
    def sample(self, num_samples: int = 1) -> Union['vector', List['vector']]:
        """
        Sample from discrete Gaussian over Z^n.
        
        EXAMPLES::
        
            sage: sampler = IdentityLatticeSampler(n=5, sigma=1.5)
            sage: samples = sampler.sample(100)
            sage: len(samples) == 100
            True
        """
        return self.vec_sampler.sample(num_samples)
    
    def sample_centered_at(self, center: 'vector') -> 'vector':
        """Sample with a different center."""
        temp_sampler = DiscreteGaussianVectorSampler(
            sigma=self.sigma, center=center, n=self.n
        )
        return temp_sampler.sample()


class QaryLatticeSampler:
    """
    Discrete Gaussian sampler for q-ary lattices.
    
    A q-ary lattice has the form:
        Λ_q(A) = {x ∈ Z^n : Ax ≡ 0 (mod q)}
    
    Or its shifted version:
        Λ_q^⊥(A) = {x ∈ Z^n : A^T x ≡ 0 (mod q)}
    """
    
    def __init__(self, A: 'Matrix', q: int, sigma: float, 
                 dual: bool = False, center: Optional['vector'] = None):
        """
        Initialize q-ary lattice sampler.
        
        Args:
            A: Matrix defining the lattice
            q: Modulus
            sigma: Gaussian parameter
            dual: If True, use Λ_q^⊥(A), otherwise Λ_q(A)
            center: Center vector
            
        EXAMPLES::
        
            sage: A = matrix(ZZ, 2, 3, [1, 2, 3, 4, 5, 6])
            sage: sampler = QaryLatticeSampler(A, q=7, sigma=2.0)
        """
        self.A = A
        self.q = q
        self.sigma = RDF(sigma)
        self.dual = dual
        
        if dual:
            self.m, self.n = A.nrows(), A.ncols()
        else:
            self.n, self.m = A.nrows(), A.ncols()
            
        self.center = center if center is not None else vector(ZZ, self.n)
        
        # For q-ary lattices, we need special sampling techniques
        # This is a simplified version - full implementation would use
        # more sophisticated methods
        self._prepare_basis()
    
    def _prepare_basis(self):
        """Prepare a basis for the q-ary lattice."""
        # For Λ_q(A), a basis can be constructed as:
        # [qI_m  0  ]
        # [A    I_n ]
        # This gives a full-rank lattice in Z^(m+n)
        
        if not self.dual:
            # Construct basis for Λ_q(A)
            # Note: This is a simplified construction
            I_m = identity_matrix(ZZ, self.m)
            I_n = identity_matrix(ZZ, self.n)
            Z_mn = zero_matrix(ZZ, self.m, self.n)
            
            self.basis = block_matrix([
                [self.q * I_m, Z_mn],
                [self.A, I_n]
            ])
        else:
            # For dual lattice, use different construction
            # This is more complex and would need parity check matrix
            raise NotImplementedError("Dual q-ary lattice sampling not yet implemented")
    
    def sample(self) -> 'vector':
        """
        Sample from discrete Gaussian over q-ary lattice.
        
        ALGORITHM:
            Use the basis representation and sample from the
            transformed Gaussian distribution.
        """
        # This is a simplified implementation
        # Full version would use specialized q-ary techniques
        
        if not self.dual:
            # Sample from Z^(m+n) with appropriate covariance
            # then project to lattice
            
            # Sample continuous Gaussian
            cont_sample = vector([
                self.sigma * normalvariate(0, 1) 
                for _ in range(self.m + self.n)
            ])
            
            # Add center
            cont_sample = cont_sample + vector(list(self.center) + [0]*self.n)
            
            # Find closest lattice point using basis
            coeffs = self.basis.solve_left(cont_sample)
            rounded_coeffs = vector([round(c) for c in coeffs])
            lattice_point = self.basis * rounded_coeffs
            
            # Extract first m coordinates (the actual lattice point)
            return vector(ZZ, lattice_point[:self.m])
        else:
            raise NotImplementedError("Dual q-ary sampling not implemented")
    
    def sample_coset(self, syndrome: 'vector') -> 'vector':
        """
        Sample from coset Λ_q(A) + t where At ≡ syndrome (mod q).
        
        This is useful for applications like learning with errors (LWE).
        """
        # Find any solution t to At ≡ syndrome (mod q)
        # Then sample from Gaussian centered at t
        
        # This is a placeholder - full implementation needed
        raise NotImplementedError("Coset sampling not yet implemented")


class NTRULatticeSampler:
    """
    Discrete Gaussian sampler for NTRU lattices.
    
    Works with NTRU lattice structure to efficiently sample
    discrete Gaussians.
    """
    
    def __init__(self, ntru_lattice, sigma: float, 
                 center: Optional['vector'] = None):
        """
        Initialize NTRU lattice sampler.
        
        Args:
            ntru_lattice: NTRULattice instance (must have basis)
            sigma: Gaussian parameter
            center: Center vector (default: zero)
            
        EXAMPLES::
        
            sage: # Assuming ntru is an NTRULattice instance
            sage: # sampler = NTRULatticeSampler(ntru, sigma=100)
        """
        if not hasattr(ntru_lattice, 'basis_matrix') or ntru_lattice.basis_matrix is None:
            raise ValueError("NTRU lattice must have generated basis")
            
        self.ntru = ntru_lattice
        self.n = ntru_lattice.n
        self.dim = 2 * self.n  # NTRU lattices have dimension 2n
        self.sigma = RDF(sigma)
        self.center = center if center is not None else vector(ZZ, self.dim)
        
        # Get basis
        self.basis = ntru_lattice.basis_matrix
    
    def sample(self) -> 'vector':
        """
        Sample from discrete Gaussian over NTRU lattice.
        
        Returns:
            Lattice vector from D_{Λ,σ,c}
            
        ALGORITHM:
            1. Sample continuous Gaussian y ~ N(c, σ²I)
            2. Solve CVP to find closest lattice point
            3. Use rejection sampling if needed for exactness
        """
        # Sample continuous Gaussian
        y = vector([
            self.center[i] + self.sigma * normalvariate(0, 1)
            for i in range(self.dim)
        ])
        
        # Find closest lattice point (Babai's algorithm)
        coeffs = self.basis.solve_left(y)
        rounded_coeffs = vector([round(c) for c in coeffs])
        v = self.basis.transpose() * rounded_coeffs
        
        # For exact sampling, we should use rejection sampling
        # This is a simplified version
        return vector(ZZ, v)
    
    def sample_klein(self) -> 'vector':
        """
        Sample using Klein's algorithm (GPV sampler).
        
        This would implement the full Klein/GPV sampler for
        exact discrete Gaussian sampling.
        """
        # Placeholder for Klein's algorithm
        # Would require Gram-Schmidt orthogonalization and
        # sequential conditional sampling
        raise NotImplementedError("Klein sampler not yet implemented")
    
    def sample_short_vector(self) -> 'vector':
        """
        Sample a short vector from the NTRU lattice.
        
        Uses discrete Gaussian with appropriate parameter to
        get short vectors with high probability.
        """
        # Use smaller sigma for shorter vectors
        short_sigma = self.sigma / 2
        
        # Sample continuous Gaussian with smaller parameter
        y = vector([short_sigma * normalvariate(0, 1) for _ in range(self.dim)])
        
        # Find closest lattice point
        coeffs = self.basis.solve_left(y)
        rounded_coeffs = vector([round(c) for c in coeffs])
        v = self.basis.transpose() * rounded_coeffs
        
        return vector(ZZ, v)


class UnifiedLatticeSampler:
    """
    Unified interface for discrete Gaussian sampling on any lattice type.
    
    Automatically detects lattice type and uses appropriate sampler.
    """
    
    def __init__(self, lattice, sigma: float, center: Optional['vector'] = None):
        """
        Initialize unified sampler.
        
        Args:
            lattice: Lattice instance (Identity, Qary, or NTRU)
            sigma: Gaussian parameter
            center: Center vector
        """
        self.lattice = lattice
        self.sigma = sigma
        self.center = center
        
        # Detect lattice type and create appropriate sampler
        lattice_type = type(lattice).__name__
        
        if 'Identity' in lattice_type:
            self.sampler = IdentityLatticeSampler(
                n=lattice.n, sigma=sigma, center=center
            )
            self.sample = self.sampler.sample
            
        elif 'Qary' in lattice_type:
            self.sampler = QaryLatticeSampler(
                A=lattice.A, q=lattice.q, sigma=sigma, 
                dual=lattice.dual, center=center
            )
            self.sample = lambda n=1: [self.sampler.sample() for _ in range(n)] if n > 1 else self.sampler.sample()
            
        elif 'NTRU' in lattice_type:
            self.sampler = NTRULatticeSampler(
                ntru_lattice=lattice, sigma=sigma, center=center
            )
            self.sample = lambda n=1: [self.sampler.sample() for _ in range(n)] if n > 1 else self.sampler.sample()
            
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")


# Example usage and tests
if __name__ == "__main__":
    print("Gaussian Lattice Sampler Examples")
    print("=" * 50)
    
    # Test identity lattice
    print("\n1. Identity Lattice Z^5")
    id_sampler = IdentityLatticeSampler(n=5, sigma=2.0)
    v = id_sampler.sample()
    print(f"Sample: {v}")
    print(f"Norm: {float(v.norm()):.2f}")
    
    # Test with center
    center = vector([1, 2, 3, 4, 5])
    samples = id_sampler.sample_centered_at(center)
    print(f"Centered sample: {samples}")
    
    # Test q-ary lattice
    print("\n2. q-ary Lattice")
    A = matrix(ZZ, 2, 3, [1, 2, 3, 4, 5, 6])
    q = 7
    qary_sampler = QaryLatticeSampler(A, q=q, sigma=2.0)
    print(f"Matrix A:")
    print(A)
    print(f"Modulus q = {q}")
    
    try:
        v_qary = qary_sampler.sample()
        print(f"q-ary sample: {v_qary}")
        # Verify it's in the lattice: Av ≡ 0 (mod q)
        check = A * v_qary
        print(f"Av mod q = {check % q}")
    except Exception as e:
        print(f"q-ary sampling: {e}")
    
    print("\n✓ Examples completed!")