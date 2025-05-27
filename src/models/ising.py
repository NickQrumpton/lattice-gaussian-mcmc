"""Ising model and related discrete lattice models."""

import numpy as np
from typing import Optional
from ..core.lattice import Lattice


class IsingModel:
    """
    Ising model on a lattice.
    
    Energy function: H(s) = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
    where s_i ∈ {-1, +1}
    """
    
    def __init__(self, lattice: Lattice,
                 coupling: float = 1.0,
                 field: float = 0.0,
                 temperature: float = 1.0):
        """
        Initialize an Ising model.
        
        Args:
            lattice: Lattice structure
            coupling: Coupling strength J
            field: External field h
            temperature: Temperature T
        """
        self.lattice = lattice
        self.coupling = coupling
        self.field = field
        self.temperature = temperature
        self.beta = 1.0 / temperature
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of a configuration.
        
        Args:
            state: Spin configuration (+1 or -1 for each site)
            
        Returns:
            Total energy
        """
        energy = 0.0
        
        # Interaction term
        for i in range(self.lattice.size):
            for j in self.lattice.get_neighbors(i):
                if i < j:  # Count each pair once
                    energy -= self.coupling * state[i] * state[j]
        
        # Field term
        energy -= self.field * np.sum(state)
        
        return energy
    
    def local_energy(self, site: int, state: np.ndarray) -> float:
        """
        Compute local energy for flipping a single spin.
        
        Args:
            site: Site index
            state: Current spin configuration
            
        Returns:
            Energy change if spin at site is flipped
        """
        neighbors = self.lattice.get_neighbors(site)
        neighbor_sum = sum(state[j] for j in neighbors)
        
        # Energy change for flipping spin i
        delta_E = 2 * state[site] * (self.coupling * neighbor_sum + self.field)
        
        return delta_E
    
    def magnetization(self, state: np.ndarray) -> float:
        """Compute magnetization (average spin)."""
        return np.mean(state)
    
    def correlation_function(self, state: np.ndarray, 
                           site1: int, site2: int) -> float:
        """Compute two-point correlation function."""
        return state[site1] * state[site2]