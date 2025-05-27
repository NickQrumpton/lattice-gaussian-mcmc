"""
IdentityLattice: The Integer Lattice Z^n as a Baseline Test Case

Implements the integer lattice Z^n with efficient, analytically tractable methods,
serving as a fundamental baseline for testing and validating lattice algorithms.
"""

import numpy as np
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, pi, sqrt, exp, log, ln,
    identity_matrix, round, floor, ceil, jacobi_theta, infinity
)
from .base import Lattice
from typing import Optional, Union, List, Tuple


class IdentityLattice(Lattice):
    """
    The integer lattice Z^n with standard basis.
    
    This class provides efficient implementations of lattice operations
    by exploiting the simple structure of Z^n. It serves as a baseline
    for testing more complex lattice algorithms.
    
    Properties:
        - Basis: Identity matrix I_n
        - Gram-Schmidt basis: Identity matrix (already orthogonal)
        - Determinant: 1
        - Self-dual: Z^n* = Z^n
        - First minimum: »_1(Z^n) = 1
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the integer lattice Z^n.
        
        Args:
            dimension: The dimension n of the lattice
            
        Raises:
            ValueError: If dimension is not positive
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
            
        super().__init__(dimension)
        self._basis = identity_matrix(RDF, dimension)
        self._gram_schmidt_basis = self._basis  # Already orthogonal
        self._gram_schmidt_norms = [1.0] * dimension
        self._gram_schmidt_coeffs = identity_matrix(RDF, dimension)
        self._determinant = RDF(1.0)
        self._volume = RDF(1.0)
        
    def get_basis(self) -> Matrix:
        """
        Return the lattice basis matrix.
        
        For Z^n, this is the identity matrix I_n.
        
        Returns:
            Matrix: The n×n identity matrix
        """
        return self._basis
    
    def get_dimension(self) -> int:
        """
        Return the dimension of the lattice.
        
        Returns:
            int: The dimension n
        """
        return self.dimension
    
    def get_gram_schmidt(self) -> Tuple[Matrix, Matrix, List[float]]:
        """
        Return the Gram-Schmidt orthogonalization.
        
        For Z^n, the basis is already orthogonal, so this returns
        the identity matrix with unit norms.
        
        Returns:
            tuple: (gram_schmidt_basis, coefficients, norms)
                - gram_schmidt_basis: Identity matrix
                - coefficients: Identity matrix (¼_ij = ´_ij)
                - norms: List of ones
        """
        return (self._gram_schmidt_basis, 
                self._gram_schmidt_coeffs,
                self._gram_schmidt_norms)
    
    def nearest_plane(self, target: Union[vector, List, np.ndarray]) -> vector:
        """
        Find nearest lattice point using Babai's nearest plane algorithm.
        
        For Z^n, this is simply coordinate-wise rounding.
        
        Args:
            target: Target vector in R^n
            
        Returns:
            vector: Nearest lattice point (rounded coordinates)
        """
        target_vec = vector(RDF, target)
        
        # For Z^n, nearest plane is just rounding
        nearest = vector(ZZ, [round(coord) for coord in target_vec])
        
        return nearest
    
    def decode_cvp(self, target: Union[vector, List, np.ndarray], 
                   method: str = 'exact') -> vector:
        """
        Solve the Closest Vector Problem.
        
        For Z^n, CVP has an exact solution via coordinate-wise rounding.
        
        Args:
            target: Target vector in R^n
            method: Algorithm to use (ignored for Z^n, always exact)
            
        Returns:
            vector: Closest lattice point
        """
        # For Z^n, CVP is solved exactly by rounding
        return self.nearest_plane(target)
    
    def smoothing_parameter(self, epsilon: float = 2**(-53)) -> float:
        """
        Compute the smoothing parameter ·_µ(Z^n).
        
        For Z^n, this has the closed form:
        ·_µ(Z^n) = sqrt(ln(2n(1+1/µ))/À)
        
        Args:
            epsilon: Smoothing parameter bound
            
        Returns:
            float: The smoothing parameter ·_µ(Z^n)
        """
        n = self.dimension
        eta = sqrt(ln(2 * n * (1 + 1/epsilon)) / pi)
        return float(eta)
    
    def gaussian_partition_function(self, sigma: float, 
                                  center: Optional[Union[vector, List]] = None) -> float:
        """
        Compute the Gaussian partition function for D_{Z^n,Ã,c}.
        
        For Z^n, this factorizes as a product of 1D partition functions:
        Z = _{i=1}^n Ñ_3(c_i, e^{-À/Ã²})
        
        Args:
            sigma: Standard deviation parameter
            center: Center vector (default: origin)
            
        Returns:
            float: Partition function value
        """
        if center is None:
            center = vector(RDF, [0] * self.dimension)
        else:
            center = vector(RDF, center)
            
        # For Z^n, partition function is product of 1D functions
        # Z_i = Ñ_3(Àc_i, e^{-À/Ã²})
        q = exp(-pi / sigma**2)
        
        partition = RDF(1.0)
        for c_i in center:
            # Jacobi theta function Ñ_3(z, q)
            z = pi * c_i  # Standard convention
            theta_val = jacobi_theta(3, z, q)
            partition *= theta_val
            
        return float(partition)
    
    def volume(self) -> float:
        """
        Return the volume of the fundamental parallelepiped.
        
        For Z^n, vol(Z^n) = 1.
        
        Returns:
            float: Always 1.0
        """
        return 1.0
    
    def is_self_dual(self) -> bool:
        """
        Check if the lattice is self-dual.
        
        Z^n is self-dual: (Z^n)* = Z^n.
        
        Returns:
            bool: Always True
        """
        return True
    
    def first_minimum(self) -> float:
        """
        Return the first minimum »_1(Z^n).
        
        The shortest nonzero vector in Z^n has length 1.
        
        Returns:
            float: Always 1.0
        """
        return 1.0
    
    def theta_series(self, q: Union[float, complex], terms: int = 100) -> float:
        """
        Compute the theta series of Z^n.
        
        For Z^n, the theta series is:
        ˜_{Z^n}(q) = Ñ_3(0, q)^n
        
        Args:
            q: Parameter |q| < 1
            terms: Number of terms in series expansion (for stability)
            
        Returns:
            float: Value of theta series
        """
        if abs(q) >= 1:
            raise ValueError("Theta series requires |q| < 1")
            
        # For Z^n: ˜(q) = [Ñ_3(0, q)]^n
        theta_1d = jacobi_theta(3, 0, q)
        theta_nd = theta_1d ** self.dimension
        
        return float(theta_nd)
    
    def direct_sample(self, sigma: float, 
                     center: Optional[Union[vector, List, np.ndarray]] = None,
                     n_samples: int = 1) -> np.ndarray:
        """
        Direct sampling from discrete Gaussian on Z^n.
        
        For Z^n, we can sample each coordinate independently from
        the 1D discrete Gaussian D_{Z,Ã,c_i}.
        
        Args:
            sigma: Standard deviation parameter
            center: Center vector (default: origin)
            n_samples: Number of samples to generate
            
        Returns:
            np.ndarray: Array of samples, shape (n_samples, dimension)
        """
        if center is None:
            center = np.zeros(self.dimension)
        else:
            center = np.array(center, dtype=float)
            
        if len(center) != self.dimension:
            raise ValueError(f"Center dimension {len(center)} != lattice dimension {self.dimension}")
            
        # Sample each coordinate independently
        samples = np.zeros((n_samples, self.dimension), dtype=int)
        
        for i in range(self.dimension):
            # Sample from D_{Z,Ã,c_i}
            samples[:, i] = self._sample_1d_discrete_gaussian(
                sigma, center[i], n_samples
            )
            
        return samples
    
    def _sample_1d_discrete_gaussian(self, sigma: float, center: float, 
                                   n_samples: int) -> np.ndarray:
        """
        Sample from 1D discrete Gaussian D_{Z,Ã,c}.
        
        Args:
            sigma: Standard deviation
            center: Center
            n_samples: Number of samples
            
        Returns:
            np.ndarray: Integer samples
        """
        # Determine range to consider (tail bound)
        tail_bound = max(10, 6 * sigma)  # Very conservative
        lower = int(floor(center - tail_bound))
        upper = int(ceil(center + tail_bound))
        
        # Compute probabilities
        integers = np.arange(lower, upper + 1)
        log_probs = -(integers - center)**2 / (2 * sigma**2)
        log_probs -= np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        
        # Sample
        samples = np.random.choice(integers, size=n_samples, p=probs)
        
        return samples
    
    def covering_radius(self) -> float:
        """
        Return the covering radius of Z^n.
        
        For Z^n, the covering radius is sqrt(n)/2.
        
        Returns:
            float: The covering radius
        """
        return float(sqrt(self.dimension) / 2)
    
    def gaussian_heuristic(self) -> float:
        """
        Compute the Gaussian heuristic for Z^n.
        
        For Z^n: Ã_GH = sqrt(n/(2Àe))
        
        Returns:
            float: Gaussian heuristic value
        """
        from sage.all import e
        n = self.dimension
        sigma_gh = sqrt(n / (2 * pi * e))
        return float(sigma_gh)
    
    def successive_minima(self, k: Optional[int] = None) -> List[float]:
        """
        Return the successive minima »_i(Z^n).
        
        For Z^n, we have:
        - »_1 = »_2 = ... = »_n = 1 (basis vectors)
        - »_{n+1} = »_{n+2} = ... = »_{2n} = sqrt(2) (face diagonals)
        - etc.
        
        Args:
            k: Number of minima to compute (default: n)
            
        Returns:
            List[float]: The first k successive minima
        """
        if k is None:
            k = self.dimension
            
        minima = []
        for i in range(1, k + 1):
            if i <= self.dimension:
                # First n minima are all 1 (standard basis vectors)
                minima.append(1.0)
            else:
                # Higher minima involve vectors with multiple nonzero coords
                # This is a simplified approximation
                num_nonzero = ((i - 1) // self.dimension) + 1
                minima.append(float(sqrt(num_nonzero)))
                
        return minima
    
    def kissing_number(self) -> int:
        """
        Return the kissing number of Z^n.
        
        For Z^n, the kissing number is 2n (the 2n neighbors at distance 1).
        
        Returns:
            int: The kissing number 2n
        """
        return 2 * self.dimension
    
    def __str__(self) -> str:
        """String representation."""
        return f"IdentityLattice(Z^{self.dimension})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"IdentityLattice(dimension={self.dimension}, "
                f"det={self.get_determinant()}, "
                f"first_minimum={self.first_minimum()})")
    
    # Validation and testing methods
    
    def validate_implementation(self) -> Dict[str, bool]:
        """
        Run validation tests on the implementation.
        
        Returns:
            Dict[str, bool]: Test results
        """
        tests = {}
        
        # Test 1: Basis is identity
        tests['basis_is_identity'] = np.allclose(
            self.get_basis(), np.eye(self.dimension)
        )
        
        # Test 2: CVP gives integer rounding
        target = vector(RDF, [1.7, -2.3, 3.9][:self.dimension])
        cvp_result = self.decode_cvp(target)
        expected = vector(ZZ, [2, -2, 4][:self.dimension])
        tests['cvp_is_rounding'] = cvp_result == expected
        
        # Test 3: Volume is 1
        tests['volume_is_one'] = abs(self.volume() - 1.0) < 1e-10
        
        # Test 4: First minimum is 1
        tests['first_minimum_is_one'] = abs(self.first_minimum() - 1.0) < 1e-10
        
        # Test 5: Is self-dual
        tests['is_self_dual'] = self.is_self_dual()
        
        # Test 6: Smoothing parameter formula
        eta = self.smoothing_parameter(0.01)
        n = self.dimension
        expected_eta = float(sqrt(ln(2 * n * 101) / pi))
        tests['smoothing_parameter_correct'] = abs(eta - expected_eta) < 1e-10
        
        return tests
    

# Example usage and tests
if __name__ == '__main__':
    # Create Z^4
    lattice = IdentityLattice(4)
    
    print(f"Created: {lattice}")
    print(f"Dimension: {lattice.get_dimension()}")
    print(f"Volume: {lattice.volume()}")
    print(f"First minimum: {lattice.first_minimum()}")
    print(f"Is self-dual: {lattice.is_self_dual()}")
    print(f"Smoothing parameter (µ=0.01): {lattice.smoothing_parameter(0.01):.6f}")
    
    # Test CVP
    target = [1.7, -2.3, 3.1, -0.8]
    closest = lattice.decode_cvp(target)
    print(f"\nCVP({target}) = {closest}")
    
    # Test direct sampling
    samples = lattice.direct_sample(sigma=2.0, center=[0.5, 0, 0, -0.5], n_samples=5)
    print(f"\nDirect samples:\n{samples}")
    
    # Run validation
    print("\nValidation tests:")
    tests = lattice.validate_implementation()
    for test, passed in tests.items():
        print(f"  {test}: {'' if passed else ''}")