"""Conditional Autoregressive (CAR) models."""

import numpy as np
from scipy import sparse
from typing import Optional
from ..core.lattice import Lattice
from ..core.gaussian import GaussianDistribution


class ConditionalAutoregressive(GaussianDistribution):
    """
    Conditional Autoregressive model on a lattice.
    
    Implements the CAR model commonly used in spatial statistics.
    """
    
    def __init__(self, lattice: Lattice,
                 tau: float = 1.0,
                 rho: float = 0.9,
                 weight_matrix: Optional[sparse.spmatrix] = None):
        """
        Initialize a CAR model.
        
        Args:
            lattice: Lattice structure
            tau: Precision parameter
            rho: Spatial correlation parameter (must be in valid range)
            weight_matrix: Spatial weight matrix (if None, uses binary adjacency)
        """
        self.lattice = lattice
        self.tau = tau
        self.rho = rho
        
        # Build or use provided weight matrix
        if weight_matrix is None:
            self.W = self._build_weight_matrix()
        else:
            self.W = weight_matrix
        
        # Check validity of rho
        eigenvalues = sparse.linalg.eigsh(self.W, k=1, which='LA', return_eigenvectors=False)
        max_eigenvalue = eigenvalues[0]
        if abs(self.rho) >= 1.0 / max_eigenvalue:
            raise ValueError(f"rho must be less than 1/lambda_max = {1.0/max_eigenvalue}")
        
        # Build precision matrix: Q = tau * (I - rho * W)
        n = self.lattice.size
        I = sparse.identity(n)
        Q = self.tau * (I - self.rho * self.W)
        
        super().__init__(precision_matrix=Q)
    
    def _build_weight_matrix(self) -> sparse.csr_matrix:
        """Build binary adjacency weight matrix."""
        n = self.lattice.size
        row_ind = []
        col_ind = []
        data = []
        
        for i in range(n):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                row_ind.append(i)
                col_ind.append(j)
                data.append(1.0 / len(neighbors))  # Row-standardized weights
        
        return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    def conditional_distribution(self, site: int, 
                                current_state: np.ndarray) -> tuple:
        """
        Get conditional distribution parameters for a single site.
        
        Args:
            site: Site index
            current_state: Current state of all sites
            
        Returns:
            Tuple of (conditional_mean, conditional_variance)
        """
        # Extract relevant row of weight matrix
        w_row = self.W.getrow(site).toarray().flatten()
        
        # Conditional mean: rho * sum(w_ij * x_j)
        cond_mean = self.rho * np.dot(w_row, current_state)
        
        # Conditional variance: 1 / tau
        cond_var = 1.0 / self.tau
        
        return cond_mean, cond_var