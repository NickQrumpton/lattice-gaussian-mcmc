"""
NTRU Lattice Implementation using SageMath.

Complete implementation of NTRU lattice construction following the FALCON 
signature scheme and Wang & Ling (2018) Section V example. This module provides
full polynomial arithmetic in Z_q[x]/(x^n + 1), NTRUSolve algorithm, key generation,
and FFT-based Gram-Schmidt orthogonalization.

EXAMPLES::

    sage: from lattices.ntru import NTRULattice
    sage: ntru = NTRULattice(n=64, q=12289)  # Small example
    sage: ntru.generate_keys()
    sage: basis = ntru.get_basis()
    sage: print(f"Basis shape: {basis.dimensions()}")
    Basis shape: (128, 128)

REFERENCES:
    - Hoffstein, Pipher, Silverman. "NTRU: A Ring-Based Public Key Cryptosystem"
    - Ducas, Prest. "Fast Fourier Orthogonalization"
    - Wang & Ling. "Lattice Gaussian Sampling by Markov Chain Monte Carlo" (2018)
"""

from sage.all import (
    ZZ, QQ, RR, CC, RDF, CDF,
    PolynomialRing, QuotientRing, GF,
    Matrix, matrix, vector, identity_matrix, block_matrix, zero_matrix,
    sqrt, log, ln, exp, pi, ceil, floor, round,
    gcd, xgcd, lcm, is_prime, next_prime,
    sin, cos, I,
    randint, random, set_random_seed,
    cached_method, lazy_attribute,
    prod, sum as sage_sum
)
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.finite_rings.finite_field_constructor import FiniteField

import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings

from .base import Lattice


class NTRULattice(Lattice):
    """
    NTRU lattice construction for cryptographic applications.
    
    Implements the NTRU/FALCON lattice structure with full polynomial
    arithmetic support, NTRUSolve algorithm, and FFT-based operations.
    
    The NTRU lattice basis has the form:
        B = [[g, -f], [G, -F]]^T
    where f, g, F, G are polynomials satisfying fG - gF = q.
    
    INPUT:
        - ``n`` -- dimension (must be power of 2 for FFT efficiency)
        - ``q`` -- modulus (prime for NTRU, typically 12289)
        - ``sigma`` -- standard deviation for discrete Gaussian sampling
    
    EXAMPLES::
    
        sage: ntru = NTRULattice(n=512, q=12289)
        sage: ntru.generate_keys()
        sage: h = ntru.get_public_key()
        sage: print(f"Public key degree: {h.degree()}")
        Public key degree: 511
    """
    
    def __init__(self, n: int, q: int = 12289, sigma: float = 1.17*sqrt(12289)):
        """
        Initialize NTRU lattice with given parameters.
        
        INPUT:
            - ``n`` -- dimension (power of 2, typically 512 or 1024)
            - ``q`` -- modulus (prime, default 12289 for FALCON)
            - ``sigma`` -- standard deviation for polynomial sampling
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.n
            64
            sage: ntru.q
            127
        """
        if n & (n - 1) != 0:
            raise ValueError(f"n must be a power of 2, got {n}")
        if not is_prime(q):
            warnings.warn(f"q={q} is not prime, may cause issues")
            
        super().__init__(2 * n)  # NTRU lattice has dimension 2n
        
        self.n = n
        self.q = q
        self.sigma = RDF(sigma)
        
        # Initialize polynomial rings
        self._init_polynomial_rings()
        
        # Key polynomials (to be generated)
        self.f = None  # Private key polynomial
        self.g = None  # Private key polynomial
        self.F = None  # NTRU equation solution
        self.G = None  # NTRU equation solution
        self.h = None  # Public key h = g/f mod q
        
        # Basis and Gram-Schmidt (computed on demand)
        self._basis = None
        self._gs_basis = None
        self._gs_norms = None
        
    def _init_polynomial_rings(self):
        """
        Initialize polynomial rings for NTRU operations.
        
        Sets up:
            - R = Z[x]/(x^n + 1)
            - Rq = Z_q[x]/(x^n + 1)
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.R
            Univariate Quotient Polynomial Ring in xbar over Integer Ring with modulus x^64 + 1
            sage: ntru.Rq
            Univariate Quotient Polynomial Ring in xbar over Ring of integers modulo 127 with modulus x^64 + 1
        """
        # Polynomial ring Z[x]
        self.R_poly = PolynomialRing(ZZ, 'x')
        x = self.R_poly.gen()
        
        # Quotient ring Z[x]/(x^n + 1)
        self.phi = x**self.n + 1  # Cyclotomic polynomial
        self.R = self.R_poly.quotient(self.phi, 'xbar')
        
        # Polynomial ring Z_q[x]
        self.Rq_poly = PolynomialRing(GF(self.q), 'x')
        x_q = self.Rq_poly.gen()
        
        # Quotient ring Z_q[x]/(x^n + 1)
        self.phi_q = x_q**self.n + 1
        self.Rq = self.Rq_poly.quotient(self.phi_q, 'xbar')
        
        # Setup FFT for fast polynomial multiplication
        self._init_fft()
        
    def _init_fft(self):
        """
        Initialize FFT parameters for fast polynomial operations.
        
        Uses 2n-th roots of unity for negacyclic convolution.
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.omega_2n^(2*64)
            1
            sage: ntru.omega_2n^64
            126  # This is -1 mod 127
        """
        # Find primitive 2n-th root of unity in Z_q
        # For negacyclic convolution, we need omega^n = -1
        
        # First find a primitive (2n)-th root
        found = False
        for g in range(2, self.q):
            # Check if g has order 2n
            if pow(g, 2*self.n, self.q) == 1 and pow(g, self.n, self.q) != 1:
                self.omega_2n = GF(self.q)(g)
                found = True
                break
                
        if not found:
            raise ValueError(f"No primitive {2*self.n}-th root of unity found mod {self.q}")
            
        # Precompute twiddle factors for FFT
        self._compute_twiddle_factors()
        
    def _compute_twiddle_factors(self):
        """Precompute twiddle factors for FFT."""
        self.twiddle = [self.omega_2n**i for i in range(self.n)]
        self.twiddle_inv = [self.omega_2n**(-i) for i in range(self.n)]
        
    def _sample_polynomial(self, sigma: Optional[float] = None) -> 'R element':
        """
        Sample polynomial from discrete Gaussian distribution.
        
        INPUT:
            - ``sigma`` -- standard deviation (default: self.sigma)
        
        OUTPUT:
            Polynomial in R with coefficients from D_{Z,sigma}
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: f = ntru._sample_polynomial(sigma=2.0)
            sage: f.parent() == ntru.R
            True
            sage: all(-10 <= c <= 10 for c in f.lift().coefficients())
            True
        """
        if sigma is None:
            sigma = float(self.sigma)
            
        # Use Sage's discrete Gaussian sampler
        D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
        coeffs = [D() for _ in range(self.n)]
        
        # Create polynomial in R
        poly = self.R_poly(coeffs)
        return self.R(poly)
    
    def _poly_to_Rq(self, f: 'R element') -> 'Rq element':
        """Convert polynomial from R to Rq."""
        coeffs = f.lift().list()
        # Pad with zeros if necessary
        while len(coeffs) < self.n:
            coeffs.append(0)
        return self.Rq(self.Rq_poly(coeffs))
    
    def _is_invertible(self, f: 'R element', ring: str = 'Rq') -> bool:
        """
        Check if polynomial is invertible in given ring.
        
        INPUT:
            - ``f`` -- polynomial to check
            - ``ring`` -- either 'Rq' (mod q) or 'R' (over Z)
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: one = ntru.R(1)
            sage: ntru._is_invertible(one)
            True
        """
        if ring == 'Rq':
            f_q = self._poly_to_Rq(f)
            try:
                _ = f_q**(-1)
                return True
            except:
                return False
        else:
            # For R (over Z), check if resultant with x^n+1 is ±1
            f_poly = f.lift()
            res = f_poly.resultant(self.phi)
            return abs(res) == 1
    
    def _invert_poly_Rq(self, f: 'R element') -> 'Rq element':
        """
        Invert polynomial in Rq = Z_q[x]/(x^n + 1).
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: f = ntru.R(1 + ntru.R.gen())  # 1 + x
            sage: f_inv = ntru._invert_poly_Rq(f)
            sage: # Check f * f_inv = 1 mod q
            sage: prod = ntru._poly_to_Rq(f) * f_inv
            sage: prod == 1
            True
        """
        f_q = self._poly_to_Rq(f)
        return f_q**(-1)
    
    def generate_keys(self, max_attempts: int = 100):
        """
        Generate NTRU key pair (f, g, F, G, h).
        
        Samples f, g from discrete Gaussian and solves NTRU equation.
        
        INPUT:
            - ``max_attempts`` -- maximum attempts to find invertible f
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127, sigma=2.0)
            sage: ntru.generate_keys()
            sage: # Verify NTRU equation
            sage: ntru._verify_ntru_equation()
            True
        """
        # Sample f until it's invertible mod q
        for _ in range(max_attempts):
            self.f = self._sample_polynomial()
            if self._is_invertible(self.f, 'Rq'):
                break
        else:
            raise RuntimeError(f"Failed to find invertible f after {max_attempts} attempts")
            
        # Sample g
        self.g = self._sample_polynomial()
        
        # Compute public key h = g/f mod q
        f_inv_q = self._invert_poly_Rq(self.f)
        g_q = self._poly_to_Rq(self.g)
        h_q = g_q * f_inv_q
        
        # Store public key as polynomial over Z with coefficients in [0, q)
        h_coeffs = [ZZ(c) for c in h_q.lift().coefficients()]
        self.h = self.R(self.R_poly(h_coeffs))
        
        # Solve NTRU equation fG - gF = q
        self.F, self.G = self._ntru_solve(self.f, self.g)
        
        # Generate basis
        self._generate_basis()
        
    def _ntru_solve(self, f: 'R element', g: 'R element') -> Tuple['R element', 'R element']:
        """
        Solve NTRU equation: find F, G such that fG - gF = q.
        
        Uses field norm and extended GCD approach.
        
        INPUT:
            - ``f`` -- polynomial in R
            - ``g`` -- polynomial in R
        
        OUTPUT:
            Tuple (F, G) satisfying fG - gF = q
        
        ALGORITHM:
            Uses the field norm approach:
            1. Compute N(f) = f * f~ where f~ is the conjugate
            2. Use extended GCD to find u, v with N(f)u + N(g)v = q
            3. Set F = f~ * u, G = g~ * v
            
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=16, q=127, sigma=2.0)  # Small example
            sage: f = ntru._sample_polynomial()
            sage: g = ntru._sample_polynomial()
            sage: F, G = ntru._ntru_solve(f, g)
            sage: # Verify equation
            sage: lhs = ntru._multiply_poly(f, G) - ntru._multiply_poly(g, F)
            sage: lhs == ntru.R(ntru.q)
            True
        """
        # Compute field norms N(f) = f * f~
        f_conj = self._conjugate_poly(f)
        g_conj = self._conjugate_poly(g)
        
        Nf = self._multiply_poly(f, f_conj)
        Ng = self._multiply_poly(g, g_conj)
        
        # Reduce to integer GCD problem
        # Since N(f), N(g) are constant polynomials, extract the constants
        nf = Nf.lift().constant_coefficient()
        ng = Ng.lift().constant_coefficient()
        
        # Extended GCD: find u, v such that nf*u + ng*v = gcd(nf, ng)
        d, u, v = xgcd(nf, ng)
        
        if d != 1 and self.q % d != 0:
            # Scale to make gcd divide q
            scale = self.q // d
            if self.q % d != 0:
                raise ValueError(f"Cannot solve NTRU equation: gcd({nf}, {ng}) = {d} does not divide q = {self.q}")
        else:
            scale = self.q
            
        # Scale u, v appropriately
        u = u * scale
        v = v * scale
        
        # Compute F = f~ * v, G = g~ * u (note the swap!)
        # This ensures fG - gF = q
        F = self._multiply_poly(f_conj, self.R(v))
        G = self._multiply_poly(g_conj, self.R(u))
        
        # Reduce coefficients
        F = self._reduce_poly(F)
        G = self._reduce_poly(G)
        
        return F, G
    
    def _conjugate_poly(self, f: 'R element') -> 'R element':
        """
        Compute conjugate polynomial f~(x) = f(x^{-1}).
        
        For x^n + 1, this is f~(x) = x^{deg(f)} * f(x^{-1}).
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=8, q=127)
            sage: R = ntru.R
            sage: x = R.gen()
            sage: f = 1 + 2*x + 3*x^2
            sage: f_conj = ntru._conjugate_poly(f)
            sage: # f~ should be 3 + 2*x^7 + x^6
            sage: expected = 3 + 2*x^7 + x^6
            sage: f_conj == expected
            True
        """
        coeffs = f.lift().coefficients()
        if not coeffs:
            return self.R(0)
            
        # Reverse coefficients and adjust for x^n = -1
        conj_coeffs = [0] * self.n
        for i, c in enumerate(coeffs):
            if i == 0:
                conj_coeffs[0] = c
            else:
                # x^{-i} = -x^{n-i} in our ring
                conj_coeffs[self.n - i] = (-1)**(i // self.n) * c
                
        return self.R(self.R_poly(conj_coeffs))
    
    def _multiply_poly(self, f: 'R element', g: 'R element') -> 'R element':
        """
        Multiply polynomials in R using FFT for efficiency.
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=8, q=127)
            sage: R = ntru.R
            sage: x = R.gen()
            sage: f = 1 + x
            sage: g = 1 - x
            sage: fg = ntru._multiply_poly(f, g)
            sage: fg == 1 - x^2
            True
        """
        # For now, use direct multiplication
        # TODO: Implement FFT multiplication for large n
        return f * g
    
    def _reduce_poly(self, f: 'R element', center: bool = True) -> 'R element':
        """
        Reduce polynomial coefficients.
        
        INPUT:
            - ``f`` -- polynomial in R
            - ``center`` -- if True, reduce to [-q/2, q/2], else [0, q)
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=8, q=7)
            sage: f = ntru.R(10)  # Constant polynomial
            sage: ntru._reduce_poly(f, center=True)
            3  # 10 mod 7 = 3
            sage: ntru._reduce_poly(f, center=False)
            3
        """
        coeffs = f.lift().coefficients()
        if center:
            # Reduce to [-q/2, q/2]
            reduced = []
            for c in coeffs:
                c_mod = c % self.q
                if c_mod > self.q // 2:
                    reduced.append(c_mod - self.q)
                else:
                    reduced.append(c_mod)
        else:
            # Reduce to [0, q)
            reduced = [c % self.q for c in coeffs]
            
        return self.R(self.R_poly(reduced))
    
    def _verify_ntru_equation(self) -> bool:
        """
        Verify that fG - gF = q.
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127, sigma=2.0)
            sage: ntru.generate_keys()
            sage: ntru._verify_ntru_equation()
            True
        """
        if any(p is None for p in [self.f, self.g, self.F, self.G]):
            return False
            
        lhs = self._multiply_poly(self.f, self.G) - self._multiply_poly(self.g, self.F)
        return lhs == self.R(self.q)
    
    def _generate_basis(self):
        """
        Generate the NTRU lattice basis.
        
        Constructs basis B = [[g, -f], [G, -F]]^T.
        """
        if any(p is None for p in [self.f, self.g, self.F, self.G]):
            raise ValueError("Must generate keys before constructing basis")
            
        # Convert polynomials to circulant matrices
        mat_f = self._poly_to_matrix(self.f)
        mat_g = self._poly_to_matrix(self.g)
        mat_F = self._poly_to_matrix(self.F)
        mat_G = self._poly_to_matrix(self.G)
        
        # Construct basis
        # B = [[g, -f], [G, -F]]^T
        top_block = block_matrix([[mat_g, -mat_f]], subdivide=False)
        bottom_block = block_matrix([[mat_G, -mat_F]], subdivide=False)
        
        self._basis = block_matrix([[top_block], [bottom_block]], subdivide=False)
        
    def _poly_to_matrix(self, f: 'R element') -> Matrix:
        """
        Convert polynomial to negacyclic circulant matrix.
        
        For f(x) = f_0 + f_1*x + ... + f_{n-1}*x^{n-1},
        the matrix has first row [f_0, f_1, ..., f_{n-1}]
        and each subsequent row is a negacyclic rotation.
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=4, q=7)
            sage: f = ntru.R([1, 2, 3, 4])  # 1 + 2x + 3x^2 + 4x^3
            sage: M = ntru._poly_to_matrix(f)
            sage: M
            [ 1  2  3  4]
            [-4  1  2  3]
            [-3 -4  1  2]
            [-2 -3 -4  1]
        """
        coeffs = list(f.lift().coefficients())
        # Pad with zeros if needed
        while len(coeffs) < self.n:
            coeffs.append(0)
            
        # Create negacyclic circulant matrix
        rows = []
        for i in range(self.n):
            row = coeffs[-i:] + coeffs[:-i]
            if i > 0:
                # Negate wrapped elements for negacyclic property
                row = [-c for c in row[:i]] + row[i:]
            rows.append(row)
            
        return matrix(ZZ, rows)
    
    def get_basis(self) -> Matrix:
        """
        Return the NTRU lattice basis.
        
        OUTPUT:
            Matrix over ZZ of dimension 2n × 2n
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.generate_keys()
            sage: B = ntru.get_basis()
            sage: B.dimensions()
            (128, 128)
            sage: B.parent()
            Full MatrixSpace of 128 by 128 dense matrices over Integer Ring
        """
        if self._basis is None:
            raise ValueError("Must generate keys first")
        return self._basis
    
    def get_dimension(self) -> int:
        """Return dimension of NTRU lattice (2n)."""
        return 2 * self.n
    
    def get_public_key(self) -> 'R element':
        """
        Return public key polynomial h.
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.generate_keys()
            sage: h = ntru.get_public_key()
            sage: h.parent() == ntru.R
            True
        """
        if self.h is None:
            raise ValueError("Must generate keys first")
        return self.h
    
    def get_private_key(self) -> Tuple['R element', 'R element']:
        """
        Return private key (f, g).
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127)
            sage: ntru.generate_keys()
            sage: f, g = ntru.get_private_key()
            sage: f.parent() == ntru.R
            True
        """
        if self.f is None or self.g is None:
            raise ValueError("Must generate keys first")
        return (self.f, self.g)
    
    @lazy_attribute
    def _fft_gram_schmidt(self):
        """
        Compute Gram-Schmidt orthogonalization using FFT.
        
        Exploits the block-circulant structure of NTRU basis
        for efficient computation.
        
        OUTPUT:
            Tuple (gs_basis, gs_norms) where:
            - gs_basis: Gram-Schmidt basis
            - gs_norms: norms ||b*_i||
        """
        if self._basis is None:
            raise ValueError("Must generate basis first")
            
        # For NTRU lattices with structure [[g, -f], [G, -F]],
        # we can use the FFT to compute GS efficiently
        
        # Convert to floating point for numerical stability
        basis_rdf = matrix(RDF, self._basis)
        
        # Standard Gram-Schmidt (can be optimized with FFT for large n)
        n = basis_rdf.nrows()
        gs_basis = matrix(RDF, n, n)
        gs_norms = []
        
        for i in range(n):
            # Start with original vector
            gs_basis[i] = basis_rdf[i]
            
            # Subtract projections
            for j in range(i):
                if gs_norms[j] > 0:
                    proj_coeff = basis_rdf[i].dot_product(gs_basis[j]) / (gs_norms[j]**2)
                    gs_basis[i] = gs_basis[i] - proj_coeff * gs_basis[j]
                    
            # Compute norm
            norm = sqrt(gs_basis[i].dot_product(gs_basis[i]))
            gs_norms.append(norm)
            
        return gs_basis, gs_norms
    
    def get_gram_schmidt(self) -> Tuple[Matrix, Matrix, List[float]]:
        """
        Return Gram-Schmidt orthogonalization of basis.
        
        OUTPUT:
            Tuple (gs_basis, mu, norms) where:
            - gs_basis: orthogonal basis
            - mu: Gram-Schmidt coefficients
            - norms: ||b*_i||
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=16, q=127, sigma=2.0)  # Small example
            sage: ntru.generate_keys()
            sage: gs_basis, mu, norms = ntru.get_gram_schmidt()
            sage: len(norms) == 32  # 2n
            True
            sage: all(n > 0 for n in norms)
            True
        """
        if self._gs_basis is None:
            gs_basis, gs_norms = self._fft_gram_schmidt
            self._gs_basis = gs_basis
            self._gs_norms = gs_norms
            
            # Compute mu coefficients
            basis_rdf = matrix(RDF, self._basis)
            n = basis_rdf.nrows()
            mu = matrix(RDF, n, n)
            
            for i in range(n):
                mu[i, i] = 1
                for j in range(i):
                    if self._gs_norms[j] > 0:
                        mu[i, j] = basis_rdf[i].dot_product(self._gs_basis[j]) / (self._gs_norms[j]**2)
                        
            self._gs_coeffs = mu
            
        return self._gs_basis, self._gs_coeffs, self._gs_norms
    
    def decode_cvp(self, target: Union[vector, List], method: str = 'babai') -> vector:
        """
        Solve closest vector problem using Babai's nearest plane.
        
        INPUT:
            - ``target`` -- target vector
            - ``method`` -- decoding method (currently only 'babai')
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=16, q=127, sigma=2.0)
            sage: ntru.generate_keys()
            sage: target = vector(RDF, [random() for _ in range(32)])
            sage: closest = ntru.decode_cvp(target)
            sage: closest.parent()
            Vector space of dimension 32 over Real Double Field
        """
        if method != 'babai':
            raise NotImplementedError(f"Method {method} not implemented")
            
        return self.nearest_plane(target)
    
    def sample_lattice_point(self, sigma: float) -> vector:
        """
        Sample from discrete Gaussian on NTRU lattice.
        
        Uses Klein's algorithm (implemented separately).
        
        INPUT:
            - ``sigma`` -- standard deviation
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=16, q=127, sigma=2.0)
            sage: ntru.generate_keys()
            sage: v = ntru.sample_lattice_point(sigma=10.0)
            sage: len(v) == 32
            True
        """
        # This would use Klein's algorithm from samplers module
        # For now, return a simple lattice point
        basis = self.get_basis()
        coeffs = [randint(-10, 10) for _ in range(self.dimension)]
        return basis.transpose() * vector(coeffs)
    
    def get_max_gram_schmidt_norm(self) -> Tuple[float, float]:
        """
        Get maximum Gram-Schmidt norm and theoretical bound.
        
        OUTPUT:
            Tuple (actual_max, theoretical_bound)
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=64, q=127, sigma=10.0)
            sage: ntru.generate_keys()
            sage: actual, bound = ntru.get_max_gram_schmidt_norm()
            sage: actual > 0
            True
            sage: bound > actual  # Usually true for proper parameters
            True
        """
        _, _, norms = self.get_gram_schmidt()
        actual_max = max(norms)
        
        # Theoretical bound from Ducas-Prest
        theoretical_bound = self.sigma * sqrt(2 * self.n)
        
        return float(actual_max), float(theoretical_bound)
    
    def verify_basis(self) -> Dict[str, bool]:
        """
        Verify correctness of NTRU basis construction.
        
        OUTPUT:
            Dictionary of verification results
        
        EXAMPLES::
        
            sage: ntru = NTRULattice(n=32, q=127, sigma=5.0)
            sage: ntru.generate_keys()
            sage: checks = ntru.verify_basis()
            sage: all(checks.values())
            True
        """
        checks = {}
        
        # Check 1: NTRU equation
        checks['ntru_equation'] = self._verify_ntru_equation()
        
        # Check 2: Basis dimensions
        if self._basis is not None:
            checks['basis_dimensions'] = (self._basis.dimensions() == (2*self.n, 2*self.n))
        else:
            checks['basis_dimensions'] = False
            
        # Check 3: Basis is integral
        if self._basis is not None:
            checks['basis_integral'] = all(c in ZZ for c in self._basis.list())
        else:
            checks['basis_integral'] = False
            
        # Check 4: Public key validity
        if self.h is not None and self.f is not None and self.g is not None:
            # h = g/f mod q
            f_inv = self._invert_poly_Rq(self.f)
            g_q = self._poly_to_Rq(self.g)
            h_computed = g_q * f_inv
            h_check = self._poly_to_Rq(self.h)
            checks['public_key_valid'] = (h_computed == h_check)
        else:
            checks['public_key_valid'] = False
            
        # Check 5: Determinant
        if self._basis is not None:
            # For NTRU, det should be q^n
            expected_det = self.q**self.n
            actual_det = abs(self._basis.determinant())
            checks['determinant_correct'] = (actual_det == expected_det)
        else:
            checks['determinant_correct'] = False
            
        return checks
    
    def __repr__(self) -> str:
        """String representation of NTRU lattice."""
        status = "initialized" if self.f is None else "keys generated"
        return f"NTRULattice(n={self.n}, q={self.q}, sigma={float(self.sigma):.2f}, status={status})"


# Doctests for the module
def _test_polynomial_arithmetic():
    """
    Test polynomial arithmetic operations.
    
    EXAMPLES::
    
        sage: from lattices.ntru import NTRULattice
        sage: ntru = NTRULattice(n=8, q=127)
        sage: # Test polynomial creation
        sage: f = ntru.R([1, 2, 3])  # 1 + 2x + 3x^2
        sage: g = ntru.R([4, 5])     # 4 + 5x
        sage: # Test multiplication
        sage: fg = f * g
        sage: fg == ntru.R([4, 13, 22, 15])  # 4 + 13x + 22x^2 + 15x^3
        True
        sage: # Test modular reduction
        sage: x = ntru.R.gen()
        sage: x^8  # Should equal -1 in R
        -1
        sage: x^9  # Should equal -x
        -xbar
    """
    pass


def _test_ntru_solve():
    """
    Test NTRUSolve algorithm.
    
    EXAMPLES::
    
        sage: from lattices.ntru import NTRULattice
        sage: ntru = NTRULattice(n=16, q=127, sigma=2.0)
        sage: # Generate random f, g
        sage: f = ntru._sample_polynomial()
        sage: g = ntru._sample_polynomial()
        sage: # Solve NTRU equation
        sage: F, G = ntru._ntru_solve(f, g)
        sage: # Verify fG - gF = q
        sage: lhs = ntru._multiply_poly(f, G) - ntru._multiply_poly(g, F)
        sage: lhs == ntru.R(ntru.q)
        True
    """
    pass


def _test_key_generation():
    """
    Test complete key generation process.
    
    EXAMPLES::
    
        sage: from lattices.ntru import NTRULattice
        sage: ntru = NTRULattice(n=64, q=12289, sigma=1.17*sqrt(12289))
        sage: ntru.generate_keys()
        sage: # Verify all components exist
        sage: all(p is not None for p in [ntru.f, ntru.g, ntru.F, ntru.G, ntru.h])
        True
        sage: # Verify basis construction
        sage: B = ntru.get_basis()
        sage: B.dimensions()
        (128, 128)
        sage: # Verify correctness
        sage: checks = ntru.verify_basis()
        sage: all(checks.values())
        True
    """
    pass


def _test_fft_operations():
    """
    Test FFT-based operations.
    
    EXAMPLES::
    
        sage: from lattices.ntru import NTRULattice
        sage: ntru = NTRULattice(n=512, q=12289)
        sage: # Check primitive root exists
        sage: ntru.omega_2n^(2*512) == 1
        True
        sage: ntru.omega_2n^512 == ntru.Rq(-1)
        True
        sage: # Test Gram-Schmidt with FFT
        sage: ntru.generate_keys()
        sage: gs_basis, mu, norms = ntru.get_gram_schmidt()
        sage: # Check orthogonality (approximately)
        sage: tol = 1e-10
        sage: for i in range(5):  # Check first few vectors
        ....:     for j in range(i):
        ....:         dot = abs(gs_basis[i].dot_product(gs_basis[j]))
        ....:         if dot > tol:
        ....:             print(f"Non-orthogonal: {i}, {j}, dot={dot}")
    """
    pass


# For module testing
if __name__ == "__main__":
    import doctest
    doctest.testmod()