"""Gaussian distribution on lattices."""

import numpy as np
from scipy import sparse
from typing import Optional, Union


class GaussianDistribution:
    """
    Represents a Gaussian distribution on a lattice.
    
    The distribution has the form:
    p(x) ∝ exp(-0.5 * x^T * Q * x + b^T * x)
    
    where Q is the precision matrix (inverse covariance).
    """
    
    def __init__(self, precision_matrix: Union[np.ndarray, sparse.spmatrix], 
                 mean_vector: Optional[np.ndarray] = None):
        """
        Initialize a Gaussian distribution.
        
        Args:
            precision_matrix: Precision matrix Q (can be sparse)
            mean_vector: Mean vector (if None, assumes zero mean)
        """
        self.precision_matrix = precision_matrix
        self.dimension = precision_matrix.shape[0]
        
        if mean_vector is None:
            self.mean_vector = np.zeros(self.dimension)
        else:
            self.mean_vector = mean_vector
        
        # Linear term: b = Q * mu
        if sparse.issparse(self.precision_matrix):
            self.linear_term = self.precision_matrix @ self.mean_vector
        else:
            self.linear_term = np.dot(self.precision_matrix, self.mean_vector)
    
    def log_density(self, x: np.ndarray) -> float:
        """
        Compute log density at point x.
        
        Args:
            x: Point at which to evaluate log density
            
        Returns:
            Log density value
        """
        if sparse.issparse(self.precision_matrix):
            quadratic = 0.5 * np.dot(x, self.precision_matrix @ x)
        else:
            quadratic = 0.5 * np.dot(x, np.dot(self.precision_matrix, x))
        
        linear = np.dot(self.linear_term, x)
        return -quadratic + linear
    
    def gradient_log_density(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log density at point x.
        
        Args:
            x: Point at which to evaluate gradient
            
        Returns:
            Gradient vector
        """
        if sparse.issparse(self.precision_matrix):
            return self.linear_term - self.precision_matrix @ x
        else:
            return self.linear_term - np.dot(self.precision_matrix, x)