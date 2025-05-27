"""
Comprehensive convergence diagnostics for lattice Gaussian MCMC.

Following Wang & Ling (2018):
"On the Hardness of the Computational Ring-LWR Problem and its Applications"
"""

import numpy as np
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, pi, sqrt, exp, log, ln,
    binomial, floor, ceil
)
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict
from collections import defaultdict
import warnings
from scipy.stats import ks_2samp
from ..lattices.base import Lattice


class ConvergenceDiagnostics:
    """
    Comprehensive diagnostics for MCMC convergence to D_{›,Ã,c}.
    
    Provides tools for assessing convergence, mixing time estimation,
    and chain diagnostics specific to lattice Gaussian sampling.
    """
    
    def __init__(self, lattice: Lattice, sigma: float, center=None):
        """
        Initialize convergence diagnostics.
        
        Args:
            lattice: Target lattice
            sigma: Standard deviation parameter
            center: Center of distribution (default: zero)
        """
        self.lattice = lattice
        self.sigma = RDF(sigma)
        self.dimension = lattice.get_dimension()
        
        if center is None:
            self.center = vector(RDF, [0] * self.dimension)
        else:
            self.center = vector(RDF, center)
            
    def estimate_tvd(self, samples, target_distribution, n_bins=None):
        """
        Estimate Total Variation Distance between empirical and target.
        
        TVD = (1/2)£|P_emp(x) - À(x)|
        
        Args:
            samples: MCMC samples
            target_distribution: Target distribution D_{›,Ã,c}
            n_bins: Number of bins for discretization
            
        Returns:
            float: Estimated TVD
        """
        n_samples = len(samples)
        
        if n_bins is None:
            # Sturges' rule for number of bins
            n_bins = int(np.ceil(np.log2(n_samples)) + 1)
            
        # Discretize the space
        bins, bin_edges = self._create_bins(samples, n_bins)
        
        # Compute empirical distribution
        emp_dist = np.zeros(n_bins)
        for sample in samples:
            bin_idx = self._find_bin(sample, bin_edges)
            if 0 <= bin_idx < n_bins:
                emp_dist[bin_idx] += 1
        emp_dist /= n_samples
        
        # Compute target distribution on bins
        target_dist = np.zeros(n_bins)
        total_target = 0
        
        for i in range(n_bins):
            # Use bin center as representative
            bin_center = self._get_bin_center(i, bin_edges)
            target_prob = self._evaluate_target_distribution(
                bin_center, target_distribution
            )
            target_dist[i] = target_prob
            total_target += target_prob
            
        # Normalize target distribution
        if total_target > 0:
            target_dist /= total_target
            
        # Compute TVD
        tvd = 0.5 * np.sum(np.abs(emp_dist - target_dist))
        
        return float(tvd)
    
    def plot_tvd_decay(self, chain, target_distribution, 
                      checkpoints=None, filename=None):
        """
        Plot TVD decay over iterations showing exponential convergence.
        
        From equation (23): ||P^t(x,·) - À||_TV d (1-´)^t
        
        Args:
            chain: MCMC chain samples
            target_distribution: Target distribution
            checkpoints: Time points to evaluate (default: log-spaced)
            filename: Save plot to file
        """
        n_samples = len(chain)
        
        if checkpoints is None:
            # Log-spaced checkpoints
            checkpoints = np.unique(np.logspace(
                0, np.log10(n_samples), 50, dtype=int
            ))
            
        tvd_values = []
        
        for t in checkpoints:
            # Estimate TVD using samples up to time t
            tvd = self.estimate_tvd(chain[:t], target_distribution)
            tvd_values.append(tvd)
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(checkpoints, tvd_values, 'b-', linewidth=2, 
                    label='Empirical TVD')
        
        # Add theoretical bound if available
        if hasattr(self, 'theoretical_bound'):
            theoretical = [(1 - self.theoretical_bound)**t 
                          for t in checkpoints]
            plt.semilogy(checkpoints, theoretical, 'r--', linewidth=2,
                        label='Theoretical bound')
            
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Variation Distance', fontsize=12)
        plt.title('TVD Convergence to Target Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return checkpoints, tvd_values
    
    def empirical_mixing_time(self, chains: List[np.ndarray], 
                            epsilon: float = 0.25):
        """
        Estimate mixing time from multiple independent chains.
        
        Find t where max_x ||P^t(x,·) - À(·)||_TV d µ
        
        Args:
            chains: List of independent MCMC chains
            epsilon: TVD threshold
            
        Returns:
            int: Empirical mixing time
        """
        n_chains = len(chains)
        min_length = min(len(chain) for chain in chains)
        
        # Check TVD at various time points
        for t in range(1, min_length):
            max_tvd = 0
            
            # Compute TVD for each chain
            for chain in chains:
                # Use samples from time t onwards
                samples = chain[t:]
                
                # Estimate TVD against pooled distribution
                pooled = np.vstack([c[t:] for c in chains])
                tvd = self._estimate_tvd_between_samples(samples, pooled)
                
                max_tvd = max(max_tvd, tvd)
                
            # Check if mixed
            if max_tvd <= epsilon:
                return t
                
        # Not mixed within chain length
        warnings.warn(f"Chains did not mix within {min_length} iterations")
        return min_length
    
    def compare_to_theoretical(self, empirical_t_mix, theoretical_t_mix):
        """
        Compare empirical mixing time to theoretical prediction.
        
        Args:
            empirical_t_mix: Measured mixing time
            theoretical_t_mix: Theoretical mixing time
            
        Returns:
            dict: Comparison results
        """
        ratio = empirical_t_mix / theoretical_t_mix if theoretical_t_mix > 0 else float('inf')
        
        return {
            'empirical_mixing_time': empirical_t_mix,
            'theoretical_mixing_time': theoretical_t_mix,
            'ratio': ratio,
            'agreement': 'good' if 0.5 <= ratio <= 2.0 else 'poor',
            'empirical_faster': empirical_t_mix < theoretical_t_mix
        }
    
    def test_uniform_ergodicity(self, sampler, n_tests=100):
        """
        Test for uniform ergodicity (exponential convergence).
        
        Verify that convergence rate doesn't depend on starting point.
        
        Args:
            sampler: MCMC sampler
            n_tests: Number of starting points to test
            
        Returns:
            dict: Test results
        """
        # Generate diverse starting points
        starting_points = self._generate_starting_points(n_tests)
        
        convergence_rates = []
        
        for start in starting_points:
            # Run short chain from this starting point
            chain = sampler.sample(1000, initial_state=start)
            
            # Estimate convergence rate
            rate = self._estimate_convergence_rate(chain)
            convergence_rates.append(rate)
            
        # Check uniformity
        rates = np.array(convergence_rates)
        cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else float('inf')
        
        return {
            'mean_rate': float(np.mean(rates)),
            'std_rate': float(np.std(rates)),
            'coefficient_of_variation': float(cv),
            'is_uniform': cv < 0.1,  # Less than 10% variation
            'min_rate': float(np.min(rates)),
            'max_rate': float(np.max(rates))
        }
    
    def minorization_condition(self, sampler, n_samples=1000):
        """
        Find ´ such that P(x,y) e ´À(y) for all x,y.
        
        This is the minorization constant for uniform ergodicity.
        
        Args:
            sampler: MCMC sampler
            n_samples: Number of samples to test
            
        Returns:
            float: Minorization constant ´
        """
        # Sample points from the target distribution
        samples = sampler.sample(n_samples)
        
        min_ratio = float('inf')
        
        # Estimate minorization constant
        for i in range(min(100, n_samples)):
            x = samples[i]
            
            # Estimate P(x, ·) by running one step from x
            next_samples = []
            for _ in range(100):
                sampler.current_state = vector(RDF, x)
                sampler._mh_step()
                next_samples.append(np.array(sampler.current_state))
                
            # Compare to target distribution
            for y in next_samples[:10]:  # Check a few points
                p_xy = self._estimate_transition_probability(x, y, sampler)
                pi_y = self._evaluate_target_probability(y)
                
                if pi_y > 0:
                    ratio = p_xy / pi_y
                    min_ratio = min(min_ratio, ratio)
                    
        return float(min_ratio) if min_ratio < float('inf') else 0.0
    
    def importance_weight_distribution(self, samples, sampler):
        """
        Analyze distribution of importance weights w(x) = À(x)/q(x).
        
        Args:
            samples: MCMC samples
            sampler: Sampler with importance_weight method
            
        Returns:
            dict: Weight distribution statistics
        """
        weights = []
        
        for sample in samples:
            if hasattr(sampler, 'importance_weight'):
                w = sampler.importance_weight(sample)
            else:
                # Compute weight manually
                w = self._compute_importance_weight(sample, sampler)
            weights.append(w)
            
        weights = np.array(weights)
        
        return {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'cv': float(np.std(weights) / np.mean(weights)) if np.mean(weights) > 0 else float('inf'),
            'effective_sample_size': len(weights) / (1 + np.var(weights)),
            'percentiles': {
                '25%': float(np.percentile(weights, 25)),
                '50%': float(np.percentile(weights, 50)),
                '75%': float(np.percentile(weights, 75)),
                '95%': float(np.percentile(weights, 95)),
                '99%': float(np.percentile(weights, 99))
            }
        }
    
    def find_worst_starting_point(self, sampler, n_candidates=100):
        """
        Find starting point x_1 with maximum importance weight.
        
        This is the worst-case starting point for convergence.
        
        Args:
            sampler: MCMC sampler
            n_candidates: Number of candidates to test
            
        Returns:
            tuple: (worst_point, max_weight)
        """
        if hasattr(sampler, 'find_max_importance_weight'):
            return sampler.find_max_importance_weight(n_candidates)
            
        # Generate candidate starting points
        candidates = self._generate_starting_points(n_candidates)
        
        max_weight = 0
        worst_point = None
        
        for x in candidates:
            if hasattr(sampler, 'importance_weight'):
                w = sampler.importance_weight(x)
            else:
                w = self._compute_importance_weight(x, sampler)
                
            if w > max_weight:
                max_weight = w
                worst_point = x
                
        return worst_point, float(max_weight)
    
    def distance_to_mode(self, samples, lattice, center):
        """
        Track distance ||Bx - c|| over iterations.
        
        Verifies that samples concentrate near the mode.
        
        Args:
            samples: MCMC samples (as lattice points)
            lattice: Lattice object
            center: Mode/center of distribution
            
        Returns:
            dict: Distance statistics
        """
        distances = []
        center_vec = vector(RDF, center)
        
        for sample in samples:
            sample_vec = vector(RDF, sample)
            dist = sqrt((sample_vec - center_vec).dot_product(sample_vec - center_vec))
            distances.append(float(dist))
            
        distances = np.array(distances)
        
        # Theoretical expected distance for Gaussian
        expected_dist = self.sigma * sqrt(self.dimension)
        
        return {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'expected_distance': float(expected_dist),
            'relative_error': float(abs(np.mean(distances) - expected_dist) / expected_dist),
            'distances': distances
        }
    
    def gram_schmidt_decay(self, samples, lattice):
        """
        Check if shorter basis vectors are sampled more frequently.
        
        This validates the discrete Gaussian behavior.
        
        Args:
            samples: Coefficient vectors from MCMC
            lattice: Lattice object
            
        Returns:
            dict: Gram-Schmidt usage statistics
        """
        # Get Gram-Schmidt norms
        gs_norms = lattice.get_gram_schmidt_norms()
        n = len(gs_norms)
        
        # Count usage of each basis vector
        usage_counts = np.zeros(n)
        
        for sample in samples:
            # Find which basis vectors have non-zero coefficients
            for i in range(n):
                if abs(sample[i]) > 1e-10:
                    usage_counts[i] += 1
                    
        # Normalize
        usage_counts /= len(samples)
        
        # Expected usage inversely proportional to GS norm
        expected_usage = 1.0 / (np.array(gs_norms) + 1e-10)
        expected_usage /= np.sum(expected_usage)
        
        # Compute correlation
        correlation = np.corrcoef(usage_counts, expected_usage)[0, 1]
        
        return {
            'usage_counts': usage_counts,
            'expected_usage': expected_usage,
            'correlation': float(correlation),
            'gs_norms': gs_norms,
            'validates_theory': correlation > 0.7
        }
    
    def batch_means_se(self, samples, batch_size=None):
        """
        Compute standard errors using batch means method.
        
        Args:
            samples: MCMC samples
            batch_size: Size of batches (default: sqrt(n))
            
        Returns:
            dict: Standard errors for each dimension
        """
        n = len(samples)
        d = samples.shape[1] if len(samples.shape) > 1 else 1
        
        if batch_size is None:
            batch_size = int(np.sqrt(n))
            
        n_batches = n // batch_size
        
        # Compute batch means
        batch_means = []
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_mean = np.mean(samples[start:end], axis=0)
            batch_means.append(batch_mean)
            
        batch_means = np.array(batch_means)
        
        # Standard error estimation
        if d == 1:
            se = np.std(batch_means) / np.sqrt(n_batches)
            return {'standard_error': float(se)}
        else:
            se = np.std(batch_means, axis=0) / np.sqrt(n_batches)
            return {
                'standard_errors': se,
                'mean_se': float(np.mean(se)),
                'max_se': float(np.max(se))
            }
    
    def optimal_batch_size(self, samples, max_lag=None):
        """
        Determine optimal batch size using autocorrelation.
        
        Args:
            samples: MCMC samples
            max_lag: Maximum lag to consider
            
        Returns:
            int: Optimal batch size
        """
        n = len(samples)
        
        if max_lag is None:
            max_lag = min(n // 4, 1000)
            
        # Compute autocorrelation function
        autocorr = []
        for lag in range(1, max_lag):
            if len(samples.shape) > 1:
                # Multivariate: use first component
                acf = self._autocorrelation(samples[:, 0], lag)
            else:
                acf = self._autocorrelation(samples, lag)
            autocorr.append(acf)
            
            # Stop when autocorrelation becomes negative
            if acf < 0:
                break
                
        # Integrated autocorrelation time
        tau_int = 1 + 2 * sum(autocorr)
        
        # Optimal batch size is about 2 * tau_int
        optimal_size = int(2 * tau_int)
        
        # Ensure reasonable bounds
        optimal_size = max(10, min(optimal_size, n // 10))
        
        return optimal_size
    
    def comprehensive_convergence_report(self, chains, sampler):
        """
        Generate comprehensive convergence diagnostics report.
        
        Args:
            chains: List of MCMC chains
            sampler: MCMC sampler object
            
        Returns:
            dict: Complete convergence analysis
        """
        report = {
            'n_chains': len(chains),
            'chain_lengths': [len(chain) for chain in chains],
            'dimension': self.dimension
        }
        
        # Mixing time analysis
        report['empirical_mixing_time'] = self.empirical_mixing_time(chains)
        
        if hasattr(sampler, 'mixing_time'):
            theoretical = sampler.mixing_time()
            report['mixing_time_comparison'] = self.compare_to_theoretical(
                report['empirical_mixing_time'], theoretical
            )
            
        # Uniform ergodicity
        report['uniform_ergodicity'] = self.test_uniform_ergodicity(sampler)
        
        # Pool all chains after burn-in
        burn_in = report['empirical_mixing_time']
        pooled = np.vstack([chain[burn_in:] for chain in chains])
        
        # Importance weights
        report['importance_weights'] = self.importance_weight_distribution(
            pooled[:1000], sampler  # Use subset for efficiency
        )
        
        # Distance to mode
        report['distance_analysis'] = self.distance_to_mode(
            pooled, self.lattice, self.center
        )
        
        # Standard errors
        report['standard_errors'] = self.batch_means_se(pooled)
        report['optimal_batch_size'] = self.optimal_batch_size(pooled)
        
        # Minorization constant
        if hasattr(sampler, 'delta'):
            report['minorization_constant'] = sampler.delta
        else:
            report['minorization_constant'] = self.minorization_condition(sampler)
            
        return report
    
    # Helper methods
    
    def _create_bins(self, samples, n_bins):
        """Create bins for discretization."""
        # Use percentile-based bins for better coverage
        percentiles = np.linspace(0, 100, n_bins + 1)
        
        if len(samples.shape) == 1:
            bin_edges = np.percentile(samples, percentiles)
        else:
            # For multivariate, use first component
            bin_edges = np.percentile(samples[:, 0], percentiles)
            
        return np.arange(n_bins), bin_edges
    
    def _find_bin(self, sample, bin_edges):
        """Find which bin a sample belongs to."""
        if len(sample.shape) == 0:
            val = sample
        else:
            val = sample[0]  # Use first component
            
        return np.searchsorted(bin_edges[1:-1], val)
    
    def _get_bin_center(self, bin_idx, bin_edges):
        """Get center of a bin."""
        return (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
    
    def _evaluate_target_distribution(self, x, target_distribution):
        """Evaluate target distribution at a point."""
        if callable(target_distribution):
            return target_distribution(x)
        else:
            # Assume Gaussian
            diff = x - float(self.center[0])
            return exp(-diff**2 / (2 * self.sigma**2))
    
    def _evaluate_target_probability(self, x):
        """Evaluate target probability for a lattice point."""
        x_vec = vector(RDF, x)
        diff = x_vec - self.center
        return exp(-diff.dot_product(diff) / (2 * self.sigma**2))
    
    def _estimate_tvd_between_samples(self, samples1, samples2):
        """Estimate TVD between two sample sets."""
        # Use KS test as approximation
        if len(samples1.shape) > 1:
            # Use first component for multivariate
            stat, _ = ks_2samp(samples1[:, 0], samples2[:, 0])
        else:
            stat, _ = ks_2samp(samples1, samples2)
        return stat
    
    def _generate_starting_points(self, n_points):
        """Generate diverse starting points for testing."""
        points = []
        
        # Include origin
        points.append(np.zeros(self.dimension))
        
        # Random points from proposal distribution
        for _ in range(n_points - 1):
            # Random Gaussian
            point = np.random.randn(self.dimension) * float(self.sigma)
            points.append(point)
            
        return points
    
    def _estimate_convergence_rate(self, chain):
        """Estimate exponential convergence rate from a chain."""
        n = len(chain)
        
        # Compute autocorrelation at lag 1
        if len(chain.shape) > 1:
            acf = self._autocorrelation(chain[:, 0], 1)
        else:
            acf = self._autocorrelation(chain, 1)
            
        # Rate H -log(|acf|)
        if abs(acf) > 0:
            return -np.log(abs(acf))
        else:
            return float('inf')
    
    def _autocorrelation(self, x, lag):
        """Compute autocorrelation at given lag."""
        n = len(x)
        if lag >= n:
            return 0
            
        x_centered = x - np.mean(x)
        c0 = np.dot(x_centered, x_centered) / n
        c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / (n - lag)
        
        return c_lag / c0 if c0 > 0 else 0
    
    def _estimate_transition_probability(self, x, y, sampler):
        """Estimate P(x, y) for the sampler."""
        # This is a simplified estimation
        # In practice, would need sampler-specific implementation
        return 1.0 / self.dimension
    
    def _compute_importance_weight(self, x, sampler):
        """Compute importance weight when not provided by sampler."""
        # Simplified computation
        return 1.0