"""Gaussian Markov Random Field models."""

import numpy as np
from scipy import sparse
from typing import Optional, Union
from ..core.lattice import Lattice
from ..core.gaussian import GaussianDistribution


class GaussianMarkovRandomField(GaussianDistribution):
    """
    Gaussian Markov Random Field on a lattice.
    
    Implements various GMRF models with different precision structures.
    """
    
    def __init__(self, lattice: Lattice, 
                 precision_param: float = 1.0,
                 interaction_param: float = 0.1,
                 model_type: str = "first_order"):
        """
        Initialize a GMRF on a lattice.
        
        Args:
            lattice: Lattice structure
            precision_param: Diagonal precision parameter
            interaction_param: Off-diagonal interaction parameter
            model_type: Type of GMRF model ("first_order", "second_order")
        """
        self.lattice = lattice
        self.precision_param = precision_param
        self.interaction_param = interaction_param
        self.model_type = model_type
        
        # Build precision matrix
        Q = self._build_precision_matrix()
        
        # Initialize parent class with zero mean
        super().__init__(precision_matrix=Q)
    
    def _build_precision_matrix(self) -> sparse.csr_matrix:
        """Build the precision matrix for the GMRF."""
        n = self.lattice.size
        row_ind = []
        col_ind = []
        data = []
        
        # Diagonal entries
        for i in range(n):
            row_ind.append(i)
            col_ind.append(i)
            n_neighbors = len(self.lattice.get_neighbors(i))
            data.append(self.precision_param + n_neighbors * self.interaction_param)
        
        # Off-diagonal entries (symmetric)
        for i in range(n):
            for j in self.lattice.get_neighbors(i):
                if i < j:  # Only add each edge once
                    row_ind.extend([i, j])
                    col_ind.extend([j, i])
                    data.extend([-self.interaction_param, -self.interaction_param])
        
        return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    def conditional_mean_precision(self, site: int, 
                                   current_state: np.ndarray) -> tuple:
        """
        Compute conditional mean and precision for a single site.
        
        Args:
            site: Site index
            current_state: Current state of all sites
            
        Returns:
            Tuple of (conditional_mean, conditional_precision)
        """
        neighbors = self.lattice.get_neighbors(site)
        
        # Conditional precision (diagonal element)
        cond_precision = self.precision_param + len(neighbors) * self.interaction_param
        
        # Conditional mean
        neighbor_sum = sum(current_state[j] for j in neighbors)
        cond_mean = self.interaction_param * neighbor_sum / cond_precision
        
        return cond_mean, cond_precision