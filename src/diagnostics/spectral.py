"""
Spectral gap analysis tools for MCMC samplers.

Based on Section III of Wang & Ling (2018):
"On the Hardness of the Computational Ring-LWR Problem and its Applications"
"""

import numpy as np
from sage.all import (
    Matrix, vector, RR, RDF, CC, CDF, ZZ, QQ, pi, sqrt, exp, log, ln,
    jacobi_theta_3, eigenvalues, eigenvectors_right, real_part, abs,
    plot, list_plot, line, text
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class SpectralGapAnalyzer:
    """
    Analyzes spectral properties of MCMC transition matrices.
    
    Provides tools for computing theoretical and empirical spectral gaps,
    mixing times, and convergence analysis.
    """
    
    def __init__(self, sampler):
        """
        Initialize spectral gap analyzer.
        
        Args:
            sampler: MCMC sampler object (Klein, MHK, etc.)
        """
        self.sampler = sampler
        self.dimension = sampler.dimension
        
        # Cache computed values
        self._theoretical_gap = None
        self._empirical_gap = None
        self._transition_matrix = None
        
    def compute_theoretical_gap(self, lattice, sigma):
        """
        Calculate theoretical spectral gap using Lemma 1.
        
        From Lemma 1: q(x)/À(x) e ´ for all x  ›
        Therefore, ³ = ´ is a lower bound on the spectral gap.
        
        Args:
            lattice: Lattice object
            sigma: Standard deviation parameter
            
        Returns:
            float: Theoretical spectral gap ³ = ´
        """
        if hasattr(self.sampler, 'compute_delta'):
            # For MHK sampler, use its delta computation
            delta = self.sampler.compute_delta()
        else:
            # General computation using equation (12)
            delta = self._compute_delta_general(lattice, sigma)
            
        self._theoretical_gap = delta
        return float(delta)
    
    def _compute_delta_general(self, lattice, sigma):
        """
        General ´ computation for any sampler.
        
        ´ = min_{x›} q(x)/À(x)
        """
        # This is a simplified computation
        # In practice, would need to sample many points
        n = lattice.get_dimension()
        det_lattice = lattice.get_determinant()
        
        # Use Jacobi theta function approximation
        tau = exp(-2 * pi^2 * sigma^2 / det_lattice^(2/n))
        rho_lattice = jacobi_theta_3(0, tau)
        
        # Simplified bound
        delta = rho_lattice / (2 * n)
        
        return min(float(delta), 1.0)
    
    def jacobi_theta_values(self, tau_values=None):
        """
        Compute Jacobi theta function values from Table I.
        
        Ñ_3(0, e^{-ÀÄ}) for various Ä values.
        
        Args:
            tau_values: List of Ä values (default: Table I values)
            
        Returns:
            dict: Mapping of Ä to Ñ_3(0, e^{-ÀÄ})
        """
        if tau_values is None:
            # Values from Table I in the paper
            tau_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            
        theta_values = {}
        for tau in tau_values:
            # Ñ_3(0, e^{-ÀÄ})
            q = exp(-pi * tau)
            theta = jacobi_theta_3(0, q)
            theta_values[tau] = float(theta)
            
        return theta_values
    
    def construct_empirical_transition_matrix(self, samples, n_states=None):
        """
        Build empirical transition matrix from samples.
        
        Args:
            samples: Array of MCMC samples
            n_states: Number of states to discretize (default: auto)
            
        Returns:
            Matrix: Empirical transition matrix
        """
        n_samples = len(samples)
        
        if n_states is None:
            # Use sqrt(n_samples) as default discretization
            n_states = min(int(np.sqrt(n_samples)), 100)
            
        # Discretize the state space
        states, state_indices = self._discretize_states(samples, n_states)
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(n_samples - 1):
            state_i = state_indices[i]
            state_j = state_indices[i + 1]
            transition_counts[state_i][state_j] += 1
            
        # Build transition matrix
        P = Matrix(RDF, n_states, n_states)
        for i in range(n_states):
            total = sum(transition_counts[i].values())
            if total > 0:
                for j in range(n_states):
                    P[i, j] = transition_counts[i][j] / total
            else:
                # No transitions from state i observed
                P[i, i] = 1  # Self-loop
                
        self._transition_matrix = P
        return P
    
    def _discretize_states(self, samples, n_states):
        """
        Discretize continuous state space.
        
        Args:
            samples: Continuous samples
            n_states: Number of discrete states
            
        Returns:
            tuple: (state_centers, state_indices_for_samples)
        """
        # Use k-means clustering for discretization
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        state_indices = kmeans.fit_predict(samples)
        state_centers = kmeans.cluster_centers_
        
        return state_centers, state_indices
    
    def compute_second_eigenvalue(self, P=None):
        """
        Compute second largest eigenvalue magnitude.
        
        Args:
            P: Transition matrix (uses cached if None)
            
        Returns:
            float: |»_2|, second largest eigenvalue magnitude
        """
        if P is None:
            P = self._transition_matrix
            if P is None:
                raise ValueError("No transition matrix available")
                
        # Compute eigenvalues
        eigs = P.eigenvalues()
        
        # Sort by magnitude (largest first)
        eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
        
        # First eigenvalue should be 1 for stochastic matrix
        # Return magnitude of second eigenvalue
        if len(eigs_sorted) > 1:
            return float(abs(eigs_sorted[1]))
        else:
            return 0.0
    
    def empirical_spectral_gap(self, samples=None, n_states=None):
        """
        Estimate spectral gap from samples.
        
        ³ = 1 - |»_2| where »_2 is second largest eigenvalue.
        
        Args:
            samples: MCMC samples (optional if matrix cached)
            n_states: Discretization level
            
        Returns:
            float: Empirical spectral gap
        """
        if samples is not None:
            self.construct_empirical_transition_matrix(samples, n_states)
            
        lambda2 = self.compute_second_eigenvalue()
        self._empirical_gap = 1 - lambda2
        
        return self._empirical_gap
    
    def decompose_transition_matrix(self, P=None):
        """
        Implement P = G + eq^T decomposition from equation (18).
        
        Where:
        - e is the all-ones vector
        - q is the stationary distribution
        - G has spectral radius < 1
        
        Args:
            P: Transition matrix
            
        Returns:
            tuple: (G, e, q)
        """
        if P is None:
            P = self._transition_matrix
            if P is None:
                raise ValueError("No transition matrix available")
                
        n = P.nrows()
        
        # Find stationary distribution (left eigenvector for eigenvalue 1)
        # À P = À
        eigs_left = P.transpose().eigenvectors_right()
        
        # Find eigenvector corresponding to eigenvalue 1
        q = None
        for eval, evecs, mult in eigs_left:
            if abs(eval - 1) < 1e-10:
                q = evecs[0]
                break
                
        if q is None:
            raise ValueError("No stationary distribution found")
            
        # Normalize q
        q = q / sum(q)
        
        # All-ones vector
        e = vector(RDF, [1] * n)
        
        # Compute G = P - eq^T
        eq_transpose = Matrix(RDF, n, n)
        for i in range(n):
            for j in range(n):
                eq_transpose[i, j] = e[i] * q[j]
                
        G = P - eq_transpose
        
        return G, e, q
    
    def analyze_upper_triangular_structure(self, G):
        """
        Study upper triangular structure of matrix G.
        
        From the paper, G often has special structure that
        can be exploited for analysis.
        
        Args:
            G: Matrix from decomposition
            
        Returns:
            dict: Analysis results
        """
        n = G.nrows()
        
        # Check how close to upper triangular
        lower_norm = 0
        upper_norm = 0
        
        for i in range(n):
            for j in range(n):
                if i > j:
                    lower_norm += abs(G[i, j])^2
                else:
                    upper_norm += abs(G[i, j])^2
                    
        total_norm = lower_norm + upper_norm
        
        # Compute spectral radius
        eigs = G.eigenvalues()
        spectral_radius = max(abs(e) for e in eigs)
        
        return {
            'lower_triangular_ratio': float(lower_norm / total_norm) if total_norm > 0 else 0,
            'upper_triangular_ratio': float(upper_norm / total_norm) if total_norm > 0 else 1,
            'spectral_radius': float(spectral_radius),
            'is_nearly_upper_triangular': float(lower_norm / total_norm) < 0.1 if total_norm > 0 else False
        }
    
    def lower_bound_mixing_time(self, epsilon=0.25):
        """
        Lower bound on mixing time using spectral gap.
        
        t_mix(µ) e (1/³)ln(1/2µ)
        
        Args:
            epsilon: Total variation distance
            
        Returns:
            float: Lower bound on mixing time
        """
        if self._theoretical_gap is None:
            raise ValueError("Compute theoretical gap first")
            
        gamma = self._theoretical_gap
        
        if gamma > 0:
            return (1/gamma) * ln(1/(2*epsilon))
        else:
            return float('inf')
    
    def upper_bound_mixing_time(self, epsilon=0.25):
        """
        Upper bound on mixing time from theoretical results.
        
        For MHK: t_mix(µ) d -ln(µ)/ln(1-´)
        
        Args:
            epsilon: Total variation distance
            
        Returns:
            float: Upper bound on mixing time
        """
        if hasattr(self.sampler, 'mixing_time'):
            return self.sampler.mixing_time(epsilon)
        else:
            # General bound using spectral gap
            gamma = self._theoretical_gap or 0.01
            return (1/gamma) * ln(1/epsilon)
    
    def compare_spectral_gaps(self, samplers_dict):
        """
        Compare spectral gaps of different samplers.
        
        Implements comparison from Lemma 3.
        
        Args:
            samplers_dict: Dict mapping names to sampler objects
            
        Returns:
            dict: Spectral gaps for each sampler
        """
        results = {}
        
        for name, sampler in samplers_dict.items():
            analyzer = SpectralGapAnalyzer(sampler)
            
            # Get theoretical gap
            if hasattr(sampler, 'lattice_sage'):
                lattice = sampler.lattice_sage
            else:
                lattice = sampler.lattice
                
            sigma = sampler.sigma_sage if hasattr(sampler, 'sigma_sage') else sampler.sigma
            
            gap = analyzer.compute_theoretical_gap(lattice, sigma)
            results[name] = {
                'spectral_gap': gap,
                'mixing_time_bound': analyzer.upper_bound_mixing_time()
            }
            
        return results
    
    def plot_convergence_rates(self, sigma_range, lattice, filename=None):
        """
        Visualize 1/´ vs parameters.
        
        Shows how convergence rate varies with Ã.
        
        Args:
            sigma_range: Range of Ã values
            lattice: Lattice object
            filename: Save plot to file (optional)
        """
        delta_values = []
        convergence_rates = []
        
        for sigma in sigma_range:
            delta = self._compute_delta_general(lattice, sigma)
            delta_values.append(float(delta))
            convergence_rates.append(1/float(delta) if delta > 0 else float('inf'))
            
        # Create plot using matplotlib
        plt.figure(figsize=(10, 6))
        plt.semilogy(sigma_range, convergence_rates, 'b-', linewidth=2)
        plt.xlabel('Ã', fontsize=12)
        plt.ylabel('1/´ (Convergence Rate)', fontsize=12)
        plt.title('Convergence Rate vs Standard Deviation', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return delta_values, convergence_rates
    
    def rejection_sampler_spectrum(self, omega):
        """
        Analyze spectrum for rejection sampler.
        
        Implements Lemma 4: eigenvalues are {1, 1-1/É, ..., 1-1/É}
        
        Args:
            omega: Normalizing constant
            
        Returns:
            dict: Spectrum analysis
        """
        # For rejection sampler with normalizing constant É
        # Eigenvalues are: 1 (once) and 1-1/É (n-1 times)
        
        n = self.dimension
        eigenvalues = [1.0] + [1 - 1/omega] * (n - 1)
        
        # Spectral gap
        spectral_gap = 1/omega
        
        # Mixing time
        mixing_time = omega * ln(1/0.25)
        
        return {
            'eigenvalues': eigenvalues,
            'spectral_gap': spectral_gap,
            'second_eigenvalue': 1 - 1/omega,
            'mixing_time': mixing_time,
            'condition_number': omega
        }
    
    def optimal_normalizing_constant(self, samples=None, n_test=1000):
        """
        Find optimal normalizing constant É_0 = max w(x).
        
        Args:
            samples: Samples to test (optional)
            n_test: Number of test points if no samples
            
        Returns:
            float: Optimal É_0
        """
        if hasattr(self.sampler, 'find_max_importance_weight'):
            # Use sampler's method if available
            max_weight, _ = self.sampler.find_max_importance_weight(n_test)
            return max_weight
        else:
            # General approach: sample and compute weights
            if samples is None:
                # Generate test samples
                samples = self.sampler.sample(n_test)
                
            max_weight = 0
            for x in samples:
                if hasattr(self.sampler, 'importance_weight'):
                    w = self.sampler.importance_weight(x)
                else:
                    # Compute weight as À(x)/q(x)
                    w = self._compute_importance_weight(x)
                    
                max_weight = max(max_weight, w)
                
            return float(max_weight)
    
    def _compute_importance_weight(self, x):
        """
        Compute importance weight À(x)/q(x) for general sampler.
        
        Args:
            x: State
            
        Returns:
            float: Importance weight
        """
        # This is a placeholder - specific implementation
        # depends on the sampler type
        return 1.0
    
    def full_analysis_report(self, samples=None):
        """
        Generate comprehensive spectral analysis report.
        
        Args:
            samples: MCMC samples for empirical analysis
            
        Returns:
            dict: Complete analysis results
        """
        report = {
            'sampler_type': self.sampler.__class__.__name__,
            'dimension': self.dimension
        }
        
        # Theoretical analysis
        if hasattr(self.sampler, 'lattice_sage'):
            lattice = self.sampler.lattice_sage
        else:
            lattice = self.sampler.lattice
            
        sigma = self.sampler.sigma_sage if hasattr(self.sampler, 'sigma_sage') else self.sampler.sigma
        
        report['theoretical_gap'] = self.compute_theoretical_gap(lattice, sigma)
        report['mixing_time_lower_bound'] = self.lower_bound_mixing_time()
        report['mixing_time_upper_bound'] = self.upper_bound_mixing_time()
        
        # Empirical analysis if samples provided
        if samples is not None:
            report['empirical_gap'] = self.empirical_spectral_gap(samples)
            report['second_eigenvalue'] = self.compute_second_eigenvalue()
            
            # Matrix decomposition
            G, e, q = self.decompose_transition_matrix()
            report['stationary_distribution_entropy'] = -sum(float(q[i]) * ln(q[i]) 
                                                            for i in range(len(q)) 
                                                            if q[i] > 0)
            report['matrix_structure'] = self.analyze_upper_triangular_structure(G)
            
        # Jacobi theta values
        report['jacobi_theta_values'] = self.jacobi_theta_values()
        
        return report