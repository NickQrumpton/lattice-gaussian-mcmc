"""
SimpleLattice: A basic lattice implementation that accepts a basis matrix.
"""

import numpy as np
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, pi, sqrt, exp, log,
    identity_matrix, matrix
)
from .base import Lattice
from typing import Optional, Union, List, Tuple


class SimpleLattice(Lattice):
    """
    A simple lattice implementation that accepts an arbitrary basis matrix.
    
    This class provides a straightforward way to create a lattice from a basis
    matrix without the need for complex initialization procedures.
    """
    
    def __init__(self, basis: Union[np.ndarray, Matrix]):
        """
        Initialize lattice with given basis matrix.
        
        Args:
            basis: Basis matrix as numpy array or Sage matrix
        """
        if isinstance(basis, np.ndarray):
            basis = matrix(RDF, basis)
        
        dimension = basis.nrows()
        super().__init__(dimension)
        
        self._basis = basis
        self._compute_gram_schmidt()
        self._compute_dual_basis()
        
    def _compute_gram_schmidt(self):
        """Compute Gram-Schmidt orthogonalization."""
        n = self.dimension
        self._gram_schmidt_basis = matrix(RDF, n, n)
        self._gram_schmidt_coeffs = matrix(RDF, n, n)
        self._gram_schmidt_norms = []
        
        for i in range(n):
            # Start with original vector
            v = self._basis[i]
            
            # Subtract projections onto previous vectors
            for j in range(i):
                coeff = v.dot_product(self._gram_schmidt_basis[j]) / self._gram_schmidt_norms[j]**2
                self._gram_schmidt_coeffs[i, j] = coeff
                v = v - coeff * self._gram_schmidt_basis[j]
            
            self._gram_schmidt_basis[i] = v
            self._gram_schmidt_norms.append(float(v.norm()))
            self._gram_schmidt_coeffs[i, i] = 1
            
    def _compute_dual_basis(self):
        """Compute dual lattice basis."""
        gram = self._basis * self._basis.transpose()
        gram_inv = gram.inverse()
        self._dual_basis = self._basis.transpose() * gram_inv
        
    def get_basis(self) -> Matrix:
        """Return the basis matrix."""
        return self._basis
        
    def get_dimension(self) -> int:
        """Return lattice dimension."""
        return self.dimension
        
    @property
    def basis(self) -> np.ndarray:
        """Return basis as numpy array for compatibility."""
        return np.array(self._basis)
        
    @property
    def min_gram_schmidt_norm(self) -> float:
        """Return minimum Gram-Schmidt norm."""
        return min(self._gram_schmidt_norms)
        
    def get_gram_schmidt_basis(self) -> Matrix:
        """Return the Gram-Schmidt orthogonalized basis."""
        return self._gram_schmidt_basis
        
    def get_gram_schmidt_norms(self) -> List[float]:
        """Return the norms of Gram-Schmidt vectors."""
        return self._gram_schmidt_norms
        
    def get_dual_basis(self) -> Matrix:
        """Return the dual lattice basis."""
        return self._dual_basis
        
    def gram_matrix(self) -> Matrix:
        """Compute the Gram matrix B * B^T."""
        return self._basis * self._basis.transpose()
        
    def determinant(self) -> float:
        """Compute the lattice determinant."""
        if self._determinant is None:
            self._determinant = float(abs(self._basis.determinant()))
        return self._determinant
        
    def volume(self) -> float:
        """Compute the fundamental volume."""
        if self._volume is None:
            self._volume = self.determinant()
        return self._volume
        
    def decode(self, target: Union[np.ndarray, vector]) -> vector:
        """
        Decode target vector to nearest lattice point (CVP).
        
        Args:
            target: Target vector
            
        Returns:
            Nearest lattice point
        """
        if isinstance(target, np.ndarray):
            target = vector(RDF, target)
            
        # Use Babai's nearest plane algorithm
        coeffs = self._basis.solve_left(target)
        rounded_coeffs = vector(ZZ, [round(c) for c in coeffs])
        return self._basis.transpose() * rounded_coeffs
        
    def dist_from_lattice(self, point: Union[np.ndarray, vector]) -> float:
        """
        Compute distance from point to lattice.
        
        Args:
            point: Point in R^n
            
        Returns:
            Distance to nearest lattice point
        """
        if isinstance(point, np.ndarray):
            point = vector(RDF, point)
            
        nearest = self.decode(point)
        return float((point - nearest).norm())