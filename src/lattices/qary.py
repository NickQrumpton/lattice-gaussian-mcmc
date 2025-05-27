"""
QaryLattice: Versatile q-ary lattice construction and analysis.

Supports SIS, LWE, Ring-LWE, and Module-LWE applications in
lattice-based cryptography.
"""

import numpy as np
from sage.all import (
    Matrix, vector, MatrixSpace, PolynomialRing, GF, ZZ, QQ, RDF,
    identity_matrix, block_matrix, zero_matrix, random_matrix,
    sqrt, log, ln, exp, pi, ceil, floor, round,
    gcd, lcm, is_prime, next_prime,
    hermite_normal_form, smith_form
)
from .base import Lattice
from typing import Optional, Union, Tuple, List, Dict
import warnings


class QaryLattice(Lattice):
    """
    q-ary lattice for cryptographic applications.
    
    Represents lattices of the form:
    ›_q(A) = {x  Z^n : Ax a 0 (mod q)}
    
    Also supports dual/parity-check lattices:
    ›_q^¥(A) = {x  Z^m : A^T x a 0 (mod q)}
    
    Used in SIS, LWE, Ring-LWE, and Module-LWE constructions.
    """
    
    def __init__(self, A: Matrix, q: int, lattice_type: str = 'primal'):
        """
        Initialize q-ary lattice.
        
        Args:
            A: Matrix defining the lattice (over Z_q)
            q: Modulus
            lattice_type: 'primal' for ›_q(A) or 'dual' for ›_q^¥(A)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(q, (int, Integer)) or q <= 1:
            raise ValueError(f"Modulus q must be an integer > 1, got {q}")
            
        self.A = Matrix(ZZ, A)  # Ensure integer matrix
        self.q = ZZ(q)
        self.lattice_type = lattice_type
        
        # Dimensions
        self.n = self.A.nrows()  # Number of rows
        self.m = self.A.ncols()  # Number of columns
        
        # Set lattice dimension based on type
        if lattice_type == 'primal':
            # ›_q(A): vectors x such that Ax a 0 (mod q)
            # Dimension is m + n (kernel + qZ^n)
            dimension = self.m + self.n
        elif lattice_type == 'dual':
            # ›_q^¥(A): vectors x such that A^T x a 0 (mod q)
            # Dimension is m
            dimension = self.m
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
            
        super().__init__(dimension)
        
        # Compute and cache basis
        self._compute_basis()
        
        # Security parameters (computed on demand)
        self._security_classical = None
        self._security_quantum = None
        
    @classmethod
    def from_random_matrix(cls, n: int, m: int, q: int, 
                          lattice_type: str = 'primal') -> 'QaryLattice':
        """
        Generate q-ary lattice from random matrix.
        
        Args:
            n: Number of rows
            m: Number of columns
            q: Modulus
            lattice_type: 'primal' or 'dual'
            
        Returns:
            QaryLattice: Random q-ary lattice
        """
        # Generate uniformly random matrix over Z_q
        A = random_matrix(GF(q), n, m)
        A = Matrix(ZZ, A)  # Lift to integers
        
        return cls(A, q, lattice_type)
    
    @classmethod
    def from_lwe_instance(cls, A: Matrix, q: int) -> 'QaryLattice':
        """
        Construct dual lattice for LWE instance.
        
        For LWE with matrix A, we want the dual lattice ›_q^¥(A).
        
        Args:
            A: LWE matrix
            q: LWE modulus
            
        Returns:
            QaryLattice: Dual lattice for LWE
        """
        return cls(A, q, lattice_type='dual')
    
    def get_basis(self) -> Matrix:
        """
        Return lattice basis.
        
        For primal ›_q(A): basis is [qI_n | -A^T; 0 | I_m]^T
        For dual ›_q^¥(A): basis is [A | qI_m]
        
        Returns:
            Matrix: Lattice basis
        """
        return self._basis
    
    def get_dimension(self) -> int:
        """Return lattice dimension."""
        return self.dimension
    
    def _compute_basis(self):
        """
        Compute explicit lattice basis.
        
        Uses standard constructions for q-ary lattices.
        """
        if self.lattice_type == 'primal':
            # Primal lattice ›_q(A)
            # Basis: [qI_n  -A^T]
            #        [0     I_m ]
            
            # Top block: [qI_n | -A^T]
            top_left = self.q * identity_matrix(ZZ, self.n)
            top_right = -self.A.transpose()
            
            # Bottom block: [0 | I_m]
            bottom_left = zero_matrix(ZZ, self.m, self.n)
            bottom_right = identity_matrix(ZZ, self.m)
            
            # Combine blocks
            self._basis = block_matrix([
                [top_left, top_right],
                [bottom_left, bottom_right]
            ])
            
        else:  # dual
            # Dual lattice ›_q^¥(A)
            # Basis: [A | qI_m]
            self._basis = block_matrix([
                [self.A, self.q * identity_matrix(ZZ, self.m)]
            ], subdivide=False)
            
        # Convert to appropriate type
        self._basis = Matrix(RDF, self._basis)
        
    def nearest_plane_modq(self, target: Union[vector, List, np.ndarray],
                          coset: Optional[Union[vector, List]] = None) -> vector:
        """
        Babai's nearest plane with q-ary structure.
        
        Exploits the modular structure for efficient decoding.
        
        Args:
            target: Target vector
            coset: Optional coset representative
            
        Returns:
            vector: Decoded lattice vector
        """
        target_vec = vector(RDF, target)
        
        if coset is not None:
            coset_vec = vector(RDF, coset)
            target_vec = target_vec - coset_vec
            
        # Use parent class nearest plane
        result = self.nearest_plane(target_vec)
        
        if coset is not None:
            result = result + coset_vec
            
        return result
    
    def estimate_security_level(self, attack_model: str = 'bkz',
                               quantum: bool = False) -> Dict[str, float]:
        """
        Estimate security level against lattice attacks.
        
        Uses state-of-the-art attack cost models.
        
        Args:
            attack_model: Attack type ('bkz', 'sieve', 'hybrid')
            quantum: Use quantum attack costs
            
        Returns:
            dict: Security estimates in bits
        """
        # Get lattice parameters
        n = self.dimension
        q = float(self.q)
        
        # Estimate shortest vector length (Gaussian heuristic)
        det_lattice = self.get_determinant()
        lambda1_gh = sqrt(n / (2 * pi * exp(1))) * det_lattice**(1/n)
        
        # Root Hermite factor needed
        # For q-ary: »_1 H min(q, sqrt(n*q))
        if self.lattice_type == 'primal':
            lambda1_est = min(q, sqrt(self.n * q))
        else:
            lambda1_est = sqrt(self.m * q)
            
        delta = (lambda1_est / det_lattice**(1/n))**(1/n)
        
        # BKZ block size estimate
        # ´ = ((À*²)^(1/²) * ²/(2Àe))^(1/(2(²-1)))
        # Approximation: ² H n / (log(n) * log(´))
        if delta > 0 and delta < 1:
            log_delta = float(ln(delta))
            beta_est = n / (log(n) * abs(log_delta))
            beta_est = max(50, min(beta_est, n))  # Reasonable bounds
        else:
            beta_est = n
            
        # Cost models
        if attack_model == 'bkz':
            if quantum:
                # Quantum sieving in BKZ
                cost_bits = 0.265 * beta_est + 16.4  # ADPS16
            else:
                # Classical BKZ-2.0
                cost_bits = 0.292 * beta_est + 16.4  # Becker et al.
                
        elif attack_model == 'sieve':
            if quantum:
                # Grover + sieving
                cost_bits = 0.265 * beta_est
            else:
                # Classical sieving
                cost_bits = 0.292 * beta_est
                
        else:  # hybrid
            # Simplified hybrid attack estimate
            cost_bits = min(0.292 * beta_est, n/2)
            
        return {
            'security_bits': float(cost_bits),
            'block_size': float(beta_est),
            'root_hermite_factor': float(delta),
            'model': attack_model,
            'quantum': quantum
        }
    
    def gaussian_width(self) -> float:
        """
        Compute/bound the Gaussian width of the lattice.
        
        For q-ary lattices, this is related to smoothing parameter.
        
        Returns:
            float: Gaussian width estimate
        """
        # Use smoothing parameter as proxy
        smoothing = self.smoothing_parameter()
        
        # Gaussian width H sqrt(n) * ·_µ(›)
        width = sqrt(self.dimension) * smoothing
        
        return float(width)
    
    @classmethod
    def ideal_qary_lattice(cls, n: int, q: int, f: Optional[List[int]] = None,
                          ring_type: str = 'cyclotomic') -> 'QaryLattice':
        """
        Construct ideal lattice for Ring-LWE.
        
        Args:
            n: Dimension (degree of polynomial ring)
            q: Modulus
            f: Defining polynomial coefficients (default: x^n + 1)
            ring_type: 'cyclotomic' or 'general'
            
        Returns:
            QaryLattice: Ideal lattice
        """
        if f is None:
            if ring_type == 'cyclotomic':
                # Standard cyclotomic: x^n + 1
                if n & (n - 1) != 0:
                    warnings.warn(f"n={n} is not a power of 2 for cyclotomic ring")
                f = [1] + [0] * (n - 1) + [1]  # x^n + 1
            else:
                raise ValueError("Must specify polynomial f for general ring")
                
        # Construct negacyclic matrix for multiplication by x
        # This represents the ideal generated by (q, x) in R = Z[x]/(f(x))
        rot_matrix = Matrix(ZZ, n, n)
        
        if f == [1] + [0] * (n - 1) + [1]:  # x^n + 1
            # Negacyclic rotation
            for i in range(n):
                for j in range(n):
                    if j == (i + 1) % n:
                        rot_matrix[i, j] = 1
                    elif i == n - 1 and j == 0:
                        rot_matrix[i, j] = -1
        else:
            # General polynomial - use companion matrix
            # TODO: Implement general case
            raise NotImplementedError("General polynomials not yet supported")
            
        # Create block matrix [qI_n | rot_matrix]
        A = block_matrix([[q * identity_matrix(ZZ, n), rot_matrix]], 
                        subdivide=False)
        
        return cls(A, q, lattice_type='primal')
    
    @classmethod
    def module_qary_lattice(cls, n: int, k: int, q: int,
                           f: Optional[List[int]] = None) -> 'QaryLattice':
        """
        Construct module lattice for Module-LWE.
        
        Args:
            n: Ring dimension
            k: Module rank
            q: Modulus
            f: Defining polynomial (default: x^n + 1)
            
        Returns:
            QaryLattice: Module lattice
        """
        # Module lattice is k copies of ideal lattice
        # Represented as block diagonal matrix
        
        # First create single ideal lattice
        ideal_lat = cls.ideal_qary_lattice(n, q, f)
        single_block = ideal_lat.A
        
        # Create k x k block diagonal
        blocks = []
        for i in range(k):
            row_blocks = []
            for j in range(k):
                if i == j:
                    row_blocks.append(single_block)
                else:
                    row_blocks.append(zero_matrix(ZZ, n, n))
            blocks.append(row_blocks)
            
        A = block_matrix(blocks, subdivide=False)
        
        return cls(A, q, lattice_type='primal')
    
    def to_sis_instance(self) -> Dict[str, Union[Matrix, int]]:
        """
        Convert to SIS problem instance.
        
        SIS: Given A, find short x such that Ax a 0 (mod q).
        
        Returns:
            dict: SIS parameters
        """
        return {
            'matrix': self.A,
            'modulus': self.q,
            'dimension': (self.n, self.m),
            'type': 'SIS',
            'norm_bound': float(sqrt(self.m) * self.q**(self.n/self.m))
        }
    
    def to_lwe_instance(self) -> Dict[str, Union[Matrix, int]]:
        """
        Convert to LWE problem instance.
        
        LWE: Given (A, b = As + e), find s.
        
        Returns:
            dict: LWE parameters
        """
        if self.lattice_type != 'dual':
            warnings.warn("LWE typically uses dual lattice")
            
        return {
            'matrix': self.A,
            'modulus': self.q,
            'dimension': (self.n, self.m),
            'type': 'LWE',
            'secret_dimension': self.n,
            'sample_dimension': self.m
        }
    
    def hermite_normal_form(self) -> Matrix:
        """
        Compute Hermite Normal Form of the basis.
        
        Returns:
            Matrix: HNF of the lattice basis
        """
        basis_int = Matrix(ZZ, self._basis)
        hnf = basis_int.hermite_form()
        return hnf
    
    def volume(self) -> float:
        """
        Compute lattice volume.
        
        For q-ary lattices, this has a simple form.
        
        Returns:
            float: Lattice volume
        """
        if self.lattice_type == 'primal':
            # vol(›_q(A)) = q^n
            return float(self.q**self.n)
        else:
            # vol(›_q^¥(A)) = q^(m-n) if A has rank n
            rank_A = self.A.rank()
            return float(self.q**(self.m - rank_A))
    
    def get_determinant(self) -> float:
        """Override to use analytical formula."""
        return self.volume()
    
    def __str__(self) -> str:
        """String representation."""
        return (f"QaryLattice(type={self.lattice_type}, "
                f"n={self.n}, m={self.m}, q={self.q})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        security = self.estimate_security_level()
        return (f"QaryLattice(type={self.lattice_type}, "
                f"n={self.n}, m={self.m}, q={self.q}, "
                f"securityH{security['security_bits']:.0f} bits)")
    

# Utility functions for common parameter sets

def falcon_parameters(n: int) -> Dict[str, int]:
    """
    Get Falcon signature parameters.
    
    Args:
        n: Security level (256, 512, 1024)
        
    Returns:
        dict: Falcon parameters
    """
    params = {
        256: {'n': 256, 'q': 12289, 'sigma': 165.74},
        512: {'n': 512, 'q': 12289, 'sigma': 165.74},
        1024: {'n': 1024, 'q': 12289, 'sigma': 165.74}
    }
    
    if n not in params:
        raise ValueError(f"Unknown Falcon parameter set: {n}")
        
    return params[n]


def dilithium_parameters(level: int) -> Dict[str, int]:
    """
    Get Dilithium signature parameters.
    
    Args:
        level: Security level (2, 3, 5)
        
    Returns:
        dict: Dilithium parameters
    """
    params = {
        2: {'n': 256, 'k': 4, 'l': 4, 'q': 8380417},
        3: {'n': 256, 'k': 6, 'l': 5, 'q': 8380417},
        5: {'n': 256, 'k': 8, 'l': 7, 'q': 8380417}
    }
    
    if level not in params:
        raise ValueError(f"Unknown Dilithium level: {level}")
        
    return params[level]


# Example usage
if __name__ == '__main__':
    # Example 1: Random q-ary lattice
    print("Example 1: Random q-ary lattice")
    lattice = QaryLattice.from_random_matrix(n=10, m=20, q=127)
    print(f"Created: {lattice}")
    print(f"Volume: {lattice.volume()}")
    print(f"Security: {lattice.estimate_security_level()}")
    
    # Example 2: LWE lattice
    print("\nExample 2: LWE lattice")
    lwe_lattice = QaryLattice.from_random_matrix(n=256, m=512, q=3329, 
                                                lattice_type='dual')
    print(f"LWE lattice: {lwe_lattice}")
    print(f"LWE instance: {lwe_lattice.to_lwe_instance()}")
    
    # Example 3: Ring-LWE (ideal) lattice
    print("\nExample 3: Ring-LWE lattice")
    rlwe_lattice = QaryLattice.ideal_qary_lattice(n=256, q=7681)
    print(f"Ring-LWE lattice: {rlwe_lattice}")
    
    # Example 4: Module-LWE lattice
    print("\nExample 4: Module-LWE lattice")
    mlwe_lattice = QaryLattice.module_qary_lattice(n=256, k=3, q=3329)
    print(f"Module-LWE lattice: {mlwe_lattice}")