"""
Implementation of Independent Metropolis-Hastings-Klein (IMHK) algorithm.

IMHK uses Klein's algorithm as an independent proposal distribution
in a Metropolis-Hastings framework to sample exactly from the discrete
Gaussian distribution.
"""

import numpy as np
from typing import Optional, Tuple, List
import time
import logging
from .base import DiscreteGaussianSampler
from .klein import KleinSampler

logger = logging.getLogger(__name__)


class IMHKSampler(DiscreteGaussianSampler):
    """
    Independent Metropolis-Hastings-Klein sampler.
    
    This algorithm uses Klein's algorithm as a proposal distribution
    in an independent Metropolis-Hastings framework to sample exactly
    from the discrete Gaussian distribution D_{Λ,σ,c}.
    
    References:
        Z. Wang and C. Ling, "Lattice Gaussian Sampling by Markov Chain 
        Monte Carlo," IEEE Trans. Information Theory, 2019.
    """
    
    def __init__(self, lattice, sigma: float, center: Optional[np.ndarray] = None,
                 precision: int = 10, burn_in: Optional[int] = None):
        """
        Initialize IMHK sampler.
        
        Args:
            lattice: Lattice instance
            sigma: Standard deviation parameter
            center: Center of the distribution
            precision: Precision parameter for Klein's algorithm
            burn_in: Number of burn-in steps (default: auto-computed)
        """
        super().__init__(lattice, sigma, center)
        
        # Initialize Klein sampler as proposal
        self.proposal_sampler = KleinSampler(
            lattice, sigma, center, precision
        )
        
        # MCMC parameters
        self.burn_in = burn_in if burn_in is not None else self._estimate_burn_in()
        
        # Track acceptance statistics
        self.total_proposals = 0
        self.accepted_proposals = 0
        
        # Current state of the Markov chain
        self.current_state = None
        self.current_coeffs = None
        self.current_log_weight = None
        
        # Precompute partition function approximation
        self._precompute_partition_bounds()
        
        logger.info(f"Initialized IMHK with burn-in={self.burn_in}")
    
    def _estimate_burn_in(self) -> int:
        """
        Estimate burn-in time based on theoretical mixing time.
        
        From the paper: t_mix(ε) < -ln(ε) * (1/δ)
        where δ = ρ_{σ,c}(Λ) / Π_i ρ_{σ_i}(Z)
        """
        # Conservative estimate: use ε = 0.01
        epsilon = 0.01
        
        # Estimate 1/δ (upper bound)
        # This is a simplified estimate; full computation requires
        # estimating ρ_{σ,c}(Λ)
        dimension_factor = self.dimension
        sigma_factor = (self.sigma / self.lattice.min_gram_schmidt_norm) ** self.dimension
        
        # Conservative upper bound
        inv_delta_estimate = dimension_factor * sigma_factor
        
        # Mixing time estimate
        mixing_time = int(np.ceil(-np.log(epsilon) * inv_delta_estimate))
        
        # Add safety factor
        return min(mixing_time * 2, 10000)
    
    def _precompute_partition_bounds(self):
        """Precompute bounds on partition functions for efficiency."""
        # Compute Klein's partition function approximation
        self.klein_log_partition = 0.0
        for i in range(self.dimension):
            sigma_i = self.sigma / abs(self.proposal_sampler.R[i, i])
            # Use theta function approximation for large sigma
            self.klein_log_partition += 0.5 * np.log(2 * np.pi) + np.log(sigma_i)
    
    def _compute_importance_weight(self, lattice_point: np.ndarray) -> Tuple[float, float]:
        """
        Compute importance weight w(x) = π(x)/q(x).
        
        Returns:
            Tuple of (importance_weight, log_importance_weight)
        """
        # Log probability under target (discrete Gaussian)
        log_target = self.log_gaussian_weight(lattice_point)
        
        # Log probability under proposal (Klein)
        log_proposal = self.proposal_sampler.log_probability(lattice_point)
        
        # Log importance weight
        log_weight = log_target - log_proposal
        
        # Handle numerical issues
        if log_weight > 700:  # exp(700) is near overflow
            return np.inf, log_weight
        elif log_weight < -700:  # exp(-700) is near underflow
            return 0.0, log_weight
        
        return np.exp(log_weight), log_weight
    
    def _initialize_chain(self):
        """Initialize the Markov chain with a starting state."""
        if self.current_state is None:
            # Use Klein sampler to get initial state
            self.current_state = self.proposal_sampler.sample_single()
            self.current_coeffs = np.linalg.solve(
                self.lattice.basis, 
                self.current_state
            ).round().astype(int)
            
            weight, log_weight = self._compute_importance_weight(self.current_state)
            self.current_log_weight = log_weight
            
            logger.debug(f"Initialized chain at state with log weight {log_weight:.4f}")
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Perform one MCMC step.
        
        Returns:
            Tuple of (new_state, accepted)
        """
        # Ensure chain is initialized
        if self.current_state is None:
            self._initialize_chain()
        
        # Generate proposal using Klein's algorithm
        proposal = self.proposal_sampler.sample_single()
        proposal_weight, proposal_log_weight = self._compute_importance_weight(proposal)
        
        # Compute acceptance ratio
        # α = min(1, w(y)/w(x)) where w = π/q
        if self.current_log_weight == -np.inf:
            # Current state has zero weight, always accept
            acceptance_ratio = 1.0
        else:
            log_ratio = proposal_log_weight - self.current_log_weight
            acceptance_ratio = min(1.0, np.exp(log_ratio))
        
        # Accept/reject
        self.total_proposals += 1
        if np.random.rand() < acceptance_ratio:
            # Accept proposal
            self.current_state = proposal
            self.current_log_weight = proposal_log_weight
            self.accepted_proposals += 1
            accepted = True
        else:
            # Reject proposal, keep current state
            accepted = False
        
        return self.current_state.copy(), accepted
    
    def sample_single(self) -> np.ndarray:
        """
        Generate a single sample after burn-in.
        
        Returns:
            Lattice point
        """
        # Run burn-in if needed
        if self.current_state is None:
            logger.debug(f"Running burn-in for {self.burn_in} steps")
            for _ in range(self.burn_in):
                self.step()
        
        # Return current state after one step
        state, _ = self.step()
        return state
    
    def sample(self, num_samples: int = 1, thin: int = 1) -> np.ndarray:
        """
        Generate multiple samples.
        
        Args:
            num_samples: Number of samples to generate
            thin: Thinning parameter (keep every thin-th sample)
            
        Returns:
            Array of shape (num_samples, dimension)
        """
        start_time = time.time()
        
        samples = np.zeros((num_samples, self.dimension))
        
        # Generate samples with thinning
        for i in range(num_samples):
            # Run 'thin' steps between samples
            for _ in range(thin):
                state, _ = self.step()
            samples[i] = state
        
        # Update statistics
        self.stats.samples_generated += num_samples
        self.stats.time_elapsed += time.time() - start_time
        self.stats.acceptance_rate = (
            self.accepted_proposals / self.total_proposals 
            if self.total_proposals > 0 else 0.0
        )
        
        logger.debug(f"Generated {num_samples} samples with acceptance rate "
                    f"{self.stats.acceptance_rate:.2%}")
        
        return samples
    
    def run_chain(self, num_steps: int, save_every: int = 1) -> List[np.ndarray]:
        """
        Run the Markov chain for a specified number of steps.
        
        Args:
            num_steps: Total number of MCMC steps
            save_every: Save state every this many steps
            
        Returns:
            List of saved states
        """
        saved_states = []
        
        for i in range(num_steps):
            state, accepted = self.step()
            
            if i % save_every == 0:
                saved_states.append(state.copy())
        
        return saved_states
    
    def estimate_spectral_gap(self, num_samples: int = 1000) -> float:
        """
        Estimate the spectral gap δ of the Markov chain.
        
        From the paper: δ = ρ_{σ,c}(Λ) / Π_i ρ_{σ_i}(Z)
        
        Args:
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated spectral gap
        """
        # This is a Monte Carlo estimate
        # We sample from Klein and compute importance weights
        
        klein_samples = self.proposal_sampler.sample(num_samples)
        weights = []
        
        for sample in klein_samples:
            weight, _ = self._compute_importance_weight(sample)
            if weight < np.inf:
                weights.append(weight)
        
        if not weights:
            logger.warning("All importance weights were infinite")
            return 0.0
        
        # The spectral gap is related to the minimum importance weight
        # δ ≥ min_x q(x)/π(x) = 1/max_x w(x)
        max_weight = max(weights)
        spectral_gap = 1.0 / max_weight if max_weight > 0 else 0.0
        
        return spectral_gap
    
    def diagnose_convergence(self, num_samples: int = 1000) -> dict:
        """
        Run convergence diagnostics.
        
        Returns:
            Dictionary with diagnostic information
        """
        # Reset statistics
        old_stats = self.stats
        self.reset_stats()
        
        # Generate samples
        samples = self.sample(num_samples)
        
        # Compute diagnostics
        diagnostics = {
            'acceptance_rate': self.stats.acceptance_rate,
            'spectral_gap_estimate': self.estimate_spectral_gap(100),
            'empirical_mean': self.empirical_mean(samples),
            'empirical_std': np.std(samples, axis=0),
            'theoretical_std': self.sigma * np.ones(self.dimension),
            'samples_per_second': num_samples / self.stats.time_elapsed
        }
        
        # Restore old statistics
        self.stats = old_stats
        
        return diagnostics
    
    def __repr__(self) -> str:
        return (f"IMHKSampler(lattice={self.lattice.name}, "
                f"σ={self.sigma:.4f}, burn_in={self.burn_in}, "
                f"acceptance_rate={self.stats.acceptance_rate:.2%})")