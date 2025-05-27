#!/usr/bin/env python3
"""
Convergence diagnostics for MCMC samplers.

Implements various convergence metrics including Total Variation Distance (TVD),
autocorrelation functions, and spectral gap estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from scipy import stats, signal
from collections import Counter


def compute_tvd(samples1: np.ndarray, samples2: np.ndarray, 
                bins: Optional[int] = None) -> float:
    """
    Compute Total Variation Distance between two sample sets.
    
    For discrete distributions:
        TVD(P,Q) = 0.5 * Σ|P(x) - Q(x)|
    
    Args:
        samples1: First sample set
        samples2: Second sample set  
        bins: Number of bins for continuous approximation
        
    Returns:
        TVD value in [0, 1]
    """
    if samples1.ndim == 1 and samples2.ndim == 1:
        # 1D case
        if bins is None:
            # Treat as discrete
            values = np.unique(np.concatenate([samples1, samples2]))
            
            # Count frequencies
            counts1 = Counter(samples1)
            counts2 = Counter(samples2)
            
            # Normalize to probabilities
            n1, n2 = len(samples1), len(samples2)
            
            tvd = 0.0
            for val in values:
                p1 = counts1.get(val, 0) / n1
                p2 = counts2.get(val, 0) / n2
                tvd += abs(p1 - p2)
            
            return 0.5 * tvd
        else:
            # Use histogram approximation
            range_min = min(samples1.min(), samples2.min())
            range_max = max(samples1.max(), samples2.max())
            
            hist1, _ = np.histogram(samples1, bins=bins, range=(range_min, range_max))
            hist2, _ = np.histogram(samples2, bins=bins, range=(range_min, range_max))
            
            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            return 0.5 * np.abs(hist1 - hist2).sum()
    else:
        # Multivariate case - use marginal TVDs
        tvds = []
        for i in range(samples1.shape[1]):
            tvd_i = compute_tvd(samples1[:, i], samples2[:, i], bins)
            tvds.append(tvd_i)
        
        # Return average marginal TVD
        return np.mean(tvds)


def compute_autocorrelation(x: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute autocorrelation function for time series.
    
    Args:
        x: Time series data
        max_lag: Maximum lag to compute (default: len(x)//4)
        
    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    if max_lag is None:
        max_lag = len(x) // 4
    
    # Ensure 1D
    x = np.asarray(x).flatten()
    
    # Remove mean
    x = x - np.mean(x)
    
    # Compute autocorrelation using FFT for efficiency
    # Pad to next power of 2 for FFT efficiency
    n = len(x)
    padded_len = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    
    # FFT of zero-padded signal
    fft_x = np.fft.fft(x, n=padded_len)
    
    # Power spectral density
    psd = fft_x * np.conj(fft_x)
    
    # Inverse FFT gives autocorrelation
    acf = np.fft.ifft(psd).real[:n]
    
    # Normalize
    acf = acf / acf[0]
    
    return acf[:max_lag + 1]


def integrated_autocorrelation_time(x: np.ndarray, c: float = 5.0) -> float:
    """
    Compute integrated autocorrelation time.
    
    τ_int = 1 + 2 * Σ ρ(k) for k = 1 to cutoff
    
    Uses automatic windowing following Sokal (1997).
    
    Args:
        x: Time series
        c: Window parameter (typically 4-6)
        
    Returns:
        Integrated autocorrelation time
    """
    acf = compute_autocorrelation(x)
    n = len(acf)
    
    # Compute cumulative sum
    tau_int = 0.0
    window = 1
    
    for k in range(1, n):
        tau_int += 2 * acf[k]
        
        # Check window condition
        if k >= c * tau_int:
            window = k
            break
    
    return 1 + tau_int


def spectral_gap_estimate(transition_probs: np.ndarray) -> float:
    """
    Estimate spectral gap from transition probability matrix.
    
    The spectral gap is 1 - |λ_2| where λ_2 is the second largest
    eigenvalue in magnitude.
    
    Args:
        transition_probs: Transition probability matrix
        
    Returns:
        Spectral gap estimate
    """
    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(transition_probs)
    
    # Sort by magnitude
    eigenvals = sorted(eigenvals, key=abs, reverse=True)
    
    # Spectral gap
    if len(eigenvals) > 1:
        gap = 1 - abs(eigenvals[1])
    else:
        gap = 1.0
    
    return float(gap)


def gelman_rubin_statistic(chains: List[np.ndarray]) -> float:
    """
    Compute Gelman-Rubin convergence statistic (R-hat).
    
    Compares within-chain and between-chain variance.
    R-hat > 1.1 indicates lack of convergence.
    
    Args:
        chains: List of MCMC chains (each is array of samples)
        
    Returns:
        R-hat statistic
    """
    m = len(chains)  # Number of chains
    n = min(len(chain) for chain in chains)  # Length of shortest chain
    
    # Truncate chains to same length
    chains = [chain[:n] for chain in chains]
    
    # Chain means
    chain_means = [np.mean(chain) for chain in chains]
    
    # Overall mean
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n / (m - 1) * sum((mean - overall_mean)**2 for mean in chain_means)
    
    # Within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in chains])
    
    # Pooled variance estimate
    var_pooled = ((n - 1) / n) * W + (1 / n) * B
    
    # R-hat
    R_hat = np.sqrt(var_pooled / W)
    
    return float(R_hat)


def compute_ess_per_second(samples: np.ndarray, elapsed_time: float) -> float:
    """
    Compute effective sample size per second.
    
    Args:
        samples: MCMC samples
        elapsed_time: Total sampling time in seconds
        
    Returns:
        ESS per second
    """
    from .mcmc_diag import effective_sample_size
    
    ess = effective_sample_size(samples)
    return ess / elapsed_time


def wasserstein_distance(samples1: np.ndarray, samples2: np.ndarray, p: int = 2) -> float:
    """
    Compute Wasserstein distance between two sample sets.
    
    Args:
        samples1: First sample set
        samples2: Second sample set
        p: Order of Wasserstein distance (1 or 2)
        
    Returns:
        Wasserstein-p distance
    """
    from scipy.stats import wasserstein_distance as wd1
    
    if samples1.ndim == 1:
        if p == 1:
            return wd1(samples1, samples2)
        else:
            # For p=2, use optimal transport
            from scipy.spatial.distance import cdist
            
            # Compute pairwise distances
            D = cdist(samples1.reshape(-1, 1), samples2.reshape(-1, 1))
            
            # Uniform weights
            n1, n2 = len(samples1), len(samples2)
            w1 = np.ones(n1) / n1
            w2 = np.ones(n2) / n2
            
            # Solve optimal transport (simplified)
            # This is approximate - full solution would use POT library
            sorted_idx1 = np.argsort(samples1)
            sorted_idx2 = np.argsort(samples2)
            
            # Match sorted samples
            cost = 0.0
            for i in range(min(n1, n2)):
                cost += abs(samples1[sorted_idx1[i]] - samples2[sorted_idx2[i]])**p
            
            return (cost / min(n1, n2))**(1/p)
    else:
        # Multivariate - use sliced Wasserstein
        n_projections = 100
        distances = []
        
        d = samples1.shape[1]
        for _ in range(n_projections):
            # Random projection direction
            theta = np.random.randn(d)
            theta /= np.linalg.norm(theta)
            
            # Project samples
            proj1 = samples1 @ theta
            proj2 = samples2 @ theta
            
            # 1D Wasserstein
            dist = wasserstein_distance(proj1, proj2, p)
            distances.append(dist)
        
        return np.mean(distances)


def mixing_time_estimate(tvd_values: List[float], threshold: float = 0.25) -> int:
    """
    Estimate mixing time from TVD values.
    
    Mixing time is the first iteration where TVD < threshold.
    
    Args:
        tvd_values: TVD at each iteration
        threshold: TVD threshold for mixing
        
    Returns:
        Estimated mixing time (iterations)
    """
    for i, tvd in enumerate(tvd_values):
        if tvd < threshold:
            return i
    
    # Not yet mixed
    return len(tvd_values)


def batch_means_variance(x: np.ndarray, batch_size: Optional[int] = None) -> float:
    """
    Estimate variance using batch means method.
    
    Args:
        x: Time series
        batch_size: Size of batches (default: sqrt(n))
        
    Returns:
        Variance estimate accounting for autocorrelation
    """
    n = len(x)
    
    if batch_size is None:
        batch_size = int(np.sqrt(n))
    
    n_batches = n // batch_size
    
    # Compute batch means
    batch_means = []
    for i in range(n_batches):
        batch = x[i*batch_size:(i+1)*batch_size]
        batch_means.append(np.mean(batch))
    
    # Variance of batch means
    var_batch_means = np.var(batch_means, ddof=1)
    
    # Scale up by batch size
    return batch_size * var_batch_means


if __name__ == "__main__":
    # Test diagnostics
    print("Testing convergence diagnostics...")
    
    # Generate test data
    np.random.seed(42)
    
    # Two similar distributions
    samples1 = np.random.normal(0, 1, 1000)
    samples2 = np.random.normal(0.1, 1.1, 1000)
    
    # Compute TVD
    tvd = compute_tvd(samples1, samples2, bins=30)
    print(f"TVD between N(0,1) and N(0.1,1.1): {tvd:.4f}")
    
    # Autocorrelation
    ar_process = np.zeros(1000)
    for i in range(1, 1000):
        ar_process[i] = 0.8 * ar_process[i-1] + np.random.normal()
    
    acf = compute_autocorrelation(ar_process, max_lag=20)
    print(f"\nAR(1) autocorrelation at lag 1: {acf[1]:.4f} (expected ~0.8)")
    
    # Integrated autocorrelation time
    tau = integrated_autocorrelation_time(ar_process)
    print(f"Integrated autocorrelation time: {tau:.2f}")
    
    print("\n✓ Diagnostics tests completed")