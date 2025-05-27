"""Base class for MCMC samplers."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseSampler(ABC):
    """
    Abstract base class for MCMC samplers.
    
    All samplers should inherit from this class and implement
    the step() method.
    """
    
    def __init__(self, target_distribution, initial_state: Optional[np.ndarray] = None):
        """
        Initialize the sampler.
        
        Args:
            target_distribution: Target distribution to sample from
            initial_state: Initial state of the Markov chain
        """
        self.target = target_distribution
        self.dimension = target_distribution.dimension
        
        if initial_state is None:
            self.current_state = np.zeros(self.dimension)
        else:
            self.current_state = initial_state.copy()
        
        self.n_steps = 0
        self.n_accepted = 0
    
    @abstractmethod
    def step(self) -> np.ndarray:
        """
        Perform one step of the MCMC sampler.
        
        Returns:
            New state of the Markov chain
        """
        pass
    
    def sample(self, n_samples: int, burn_in: int = 0, 
               thin: int = 1) -> np.ndarray:
        """
        Generate samples from the target distribution.
        
        Args:
            n_samples: Number of samples to generate
            burn_in: Number of burn-in steps to discard
            thin: Thinning factor
            
        Returns:
            Array of samples
        """
        # Burn-in phase
        for _ in range(burn_in):
            self.step()
        
        # Sampling phase
        samples = np.zeros((n_samples, self.dimension))
        for i in range(n_samples):
            for _ in range(thin):
                self.step()
            samples[i] = self.current_state.copy()
        
        return samples
    
    @property
    def acceptance_rate(self) -> float:
        """Compute the acceptance rate."""
        if self.n_steps == 0:
            return 0.0
        return self.n_accepted / self.n_steps
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset the sampler to initial state."""
        if initial_state is None:
            self.current_state = np.zeros(self.dimension)
        else:
            self.current_state = initial_state.copy()
        
        self.n_steps = 0
        self.n_accepted = 0