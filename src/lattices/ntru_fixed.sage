#!/usr/bin/env sage
"""
Fixed NTRU Lattice Implementation.
"""

from sage.all import *
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

class NTRULattice:
    """NTRU lattice implementation."""
    
    def __init__(self, n, q, sigma=None):
        self.n = n
        self.q = q
        self.sigma = sigma if sigma is not None else 1.17 * sqrt(q).n()
        
        # Polynomial rings
        R.<x> = PolynomialRing(ZZ)
        self.R = R.quotient(x^n + 1)
        
        Rq.<x> = PolynomialRing(GF(q))
        self.Rq = Rq.quotient(x^n + 1)
        
        # Sampler
        self.dgauss = DiscreteGaussianDistributionIntegerSampler(self.sigma)
        
        # Keys
        self.f = self.g = self.F = self.G = self.h = None
        self.basis_matrix = None
    
    def _get_coeffs(self, poly, length):
        """Get coefficient list of fixed length."""
        coeffs = list(poly.lift())
        # Pad with zeros if needed
        while len(coeffs) < length:
            coeffs.append(0)
        return coeffs[:length]
    
    def _sample_poly(self):
        """Sample Gaussian polynomial."""
        return self.R([self.dgauss() for _ in range(self.n)])
    
    def _conjugate(self, f):
        """Compute f*(x) = f(-x)."""
        coeffs = self._get_coeffs(f, self.n)
        conj = [coeffs[0]]
        for i in range(1, self.n):
            conj.append(-coeffs[self.n - i])
        return self.R(conj)
    
    def _solve_ntru(self, f, g):
        """Solve fG - gF = q."""
        # Get conjugates
        f_star = self._conjugate(f)
        g_star = self._conjugate(g)
        
        # Compute norms (in Z)
        Nf = f * f_star
        Ng = g * g_star
        
        # Get constant terms
        nf = ZZ(self._get_coeffs(Nf, self.n)[0])
        ng = ZZ(self._get_coeffs(Ng, self.n)[0])
        
        # Extended GCD
        d, u, v = xgcd(nf, ng)
        
        # Scale by q
        if d == 1:
            u_q = u * self.q
            v_q = v * self.q
        else:
            u_q = (u * self.q) // d
            v_q = (v * self.q) // d
        
        # Compute F, G
        F = f_star * self.R(v_q)
        G = g_star * self.R(u_q)
        
        return F, G
    
    def generate_keys(self, max_attempts=100):
        """Generate NTRU key pair."""
        for attempt in range(max_attempts):
            # Sample f with +1 to help invertibility
            f_coeffs = [self.dgauss() for _ in range(self.n)]
            f_coeffs[0] += 1
            self.f = self.R(f_coeffs)
            
            # Check if f invertible mod q
            f_mod_q = self.Rq(f_coeffs)
            try:
                f_inv_q = f_mod_q^(-1)
            except:
                continue
            
            # Sample g
            self.g = self._sample_poly()
            
            # Compute h = g/f mod q
            g_mod_q = self.Rq(self._get_coeffs(self.g, self.n))
            h_mod_q = g_mod_q * f_inv_q
            self.h = self.R([ZZ(c) for c in self._get_coeffs(h_mod_q, self.n)])
            
            # Solve NTRU equation
            try:
                self.F, self.G = self._solve_ntru(self.f, self.g)
                
                # Verify fG - gF = q
                check = self.f * self.G - self.g * self.F
                check_coeffs = self._get_coeffs(check, self.n)
                
                # First coeff should be ±q, others zero
                if abs(check_coeffs[0]) == self.q and all(c == 0 for c in check_coeffs[1:]):
                    # Generate basis
                    self._make_basis()
                    return True
                    
            except Exception as e:
                pass
        
        return False
    
    def _poly_to_circulant(self, f):
        """Convert polynomial to negacyclic circulant matrix."""
        coeffs = self._get_coeffs(f, self.n)
        M = matrix(ZZ, self.n, self.n)
        
        for i in range(self.n):
            for j in range(self.n):
                idx = (j - i) % self.n
                if j >= i:
                    M[i,j] = coeffs[idx]
                else:
                    M[i,j] = -coeffs[idx]
        
        return M
    
    def _make_basis(self):
        """Construct NTRU basis matrix."""
        # Convert to matrices
        M_g = self._poly_to_circulant(self.g)
        M_f = self._poly_to_circulant(self.f)
        M_G = self._poly_to_circulant(self.G)
        M_F = self._poly_to_circulant(self.F)
        
        # Build basis [[g, -f], [G, -F]]^T
        self.basis_matrix = block_matrix([
            [M_g, -M_f],
            [M_G, -M_F]
        ])
    
    def gram_schmidt_norms(self):
        """Compute Gram-Schmidt norms."""
        if self.basis_matrix is None:
            return None
            
        B = self.basis_matrix.change_ring(RDF)
        gs_norms = []
        gs_vecs = []
        
        for i in range(B.nrows()):
            v = vector(B[i])
            
            # Orthogonalize against previous vectors
            for j in range(len(gs_vecs)):
                if gs_norms[j] > 0:
                    proj_coeff = v.dot_product(gs_vecs[j]) / (gs_norms[j]^2)
                    v = v - proj_coeff * gs_vecs[j]
            
            gs_vecs.append(v)
            gs_norms.append(float(v.norm()))
        
        return gs_norms


# Test the implementation
if __name__ == "__main__":
    print("Testing Fixed NTRU Implementation")
    print("=" * 40)
    
    # Test small parameters
    n, q = 16, 257
    print(f"\nParameters: n={n}, q={q}")
    
    ntru = NTRULattice(n, q)
    print(f"Sigma: {ntru.sigma:.2f}")
    
    print("\nGenerating keys...")
    success = ntru.generate_keys(max_attempts=20)
    
    if success:
        print("✓ Keys generated successfully!")
        
        # Check basis
        B = ntru.basis_matrix
        print(f"\nBasis: {B.nrows()}×{B.ncols()}")
        
        # Compute determinant
        print("Computing determinant...")
        det = abs(B.det())
        expected = q^n
        print(f"Det: {det}")
        print(f"Expected: {expected}")
        print(f"Match: {det == expected}")
        
        # Gram-Schmidt norms
        print("\nComputing Gram-Schmidt norms...")
        gs_norms = ntru.gram_schmidt_norms()
        if gs_norms:
            print(f"Min norm: {min(gs_norms):.2f}")
            print(f"Max norm: {max(gs_norms):.2f}")
            print(f"Ratio: {max(gs_norms)/min(gs_norms):.2f}")
        
        # Show key norms
        f_norm = sqrt(sum(c^2 for c in ntru._get_coeffs(ntru.f, n)))
        g_norm = sqrt(sum(c^2 for c in ntru._get_coeffs(ntru.g, n)))
        print(f"\nKey norms:")
        print(f"||f|| = {f_norm:.2f}")
        print(f"||g|| = {g_norm:.2f}")
        
    else:
        print("✗ Key generation failed")