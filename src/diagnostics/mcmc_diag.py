#!/usr/bin/env python3
"""
MCMC-specific diagnostics.

Implements effective sample size, acceptance rates, and other MCMC metrics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


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
        max_lag = min(len(x) // 4, 1000)
    
    x = np.asarray(x).flatten()
    x = x - np.mean(x)
    
    # Use numpy's correlate for efficiency
    c = np.correlate(x, x, 'full')[len(x)-1:]
    c = c / c[0]  # Normalize
    
    return c[:max_lag + 1]


def integrated_autocorrelation_time(x: np.ndarray, c: float = 5.0) -> float:
    """
    Compute integrated autocorrelation time.
    
    Args:
        x: Time series
        c: Window parameter
        
    Returns:
        Integrated autocorrelation time
    """
    acf = compute_autocorrelation(x)
    
    # Compute cumulative sum with automatic windowing
    tau_int = 0.0
    for k in range(1, len(acf)):
        tau_int += 2 * acf[k]
        if k >= c * tau_int:
            break
    
    return 1 + tau_int


def effective_sample_size(x: np.ndarray, method: str = 'autocorr') -> float:
    """
    Compute effective sample size (ESS).
    
    ESS = n / (1 + 2*sum(autocorrelations))
    
    Args:
        x: MCMC samples (can be multivariate)
        method: 'autocorr' or 'batch_means'
        
    Returns:
        Effective sample size
    """
    if x.ndim == 1:
        n = len(x)
        
        if method == 'autocorr':
            tau = integrated_autocorrelation_time(x)
            return n / tau
        else:
            # Batch means method
            batch_size = int(np.sqrt(n))
            n_batches = n // batch_size
            
            batch_means = []
            for i in range(n_batches):
                batch = x[i*batch_size:(i+1)*batch_size]
                batch_means.append(np.mean(batch))
            
            # Variance of batch means vs sample variance
            var_batch = np.var(batch_means)
            var_sample = np.var(x) / n
            
            if var_batch > 0:
                return var_sample / var_batch * n
            else:
                return n
    else:
        # Multivariate - compute ESS for each dimension
        ess_values = []
        for i in range(x.shape[1]):
            ess_i = effective_sample_size(x[:, i], method)
            ess_values.append(ess_i)
        
        # Return minimum ESS across dimensions
        return min(ess_values)


def compute_acceptance_rate(accepted: np.ndarray) -> float:
    """
    Compute MCMC acceptance rate.
    
    Args:
        accepted: Boolean array of accept/reject decisions
        
    Returns:
        Acceptance rate in [0, 1]
    """
    return np.mean(accepted)


def compute_jump_distance(samples: np.ndarray) -> np.ndarray:
    """
    Compute jump distances between consecutive samples.
    
    Args:
        samples: MCMC samples
        
    Returns:
        Array of jump distances
    """
    if samples.ndim == 1:
        jumps = np.diff(samples)
        return np.abs(jumps)
    else:
        # Multivariate - compute Euclidean distances
        diffs = np.diff(samples, axis=0)
        return np.linalg.norm(diffs, axis=1)


def diagnose_chain(samples: np.ndarray, 
                  burn_in: Optional[int] = None,
                  thin: int = 1) -> Dict[str, Any]:
    """
    Comprehensive diagnostics for MCMC chain.
    
    Args:
        samples: MCMC samples
        burn_in: Number of burn-in samples to discard
        thin: Thinning factor
        
    Returns:
        Dictionary of diagnostic metrics
    """
    # Remove burn-in
    if burn_in is not None:
        samples = samples[burn_in:]
    
    # Apply thinning
    if thin > 1:
        samples = samples[::thin]
    
    n = len(samples)
    
    # Basic statistics
    if samples.ndim == 1:
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        quantiles = np.percentile(samples, [2.5, 25, 50, 75, 97.5])
    else:
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        quantiles = None
    
    # ESS
    ess = effective_sample_size(samples)
    
    # Autocorrelation
    if samples.ndim == 1:
        acf = compute_autocorrelation(samples, max_lag=min(100, n//4))
        tau_int = integrated_autocorrelation_time(samples)
    else:
        # Use first dimension for ACF
        acf = compute_autocorrelation(samples[:, 0], max_lag=min(100, n//4))
        tau_int = integrated_autocorrelation_time(samples[:, 0])
    
    # Jump distances
    jumps = compute_jump_distance(samples)
    mean_jump = float(np.mean(jumps))
    
    diagnostics = {
        'n_samples': n,
        'mean': mean,
        'std': std,
        'ess': float(ess),
        'ess_per_sample': float(ess / n),
        'tau_int': float(tau_int),
        'mean_jump_distance': mean_jump,
        'acf_lag_1': float(acf[1]) if len(acf) > 1 else None,
        'acf_lag_10': float(acf[10]) if len(acf) > 10 else None,
    }
    
    if quantiles is not None:
        diagnostics['quantiles'] = {
            '2.5%': float(quantiles[0]),
            '25%': float(quantiles[1]),
            '50%': float(quantiles[2]),
            '75%': float(quantiles[3]),
            '97.5%': float(quantiles[4])
        }
    
    return diagnostics


def compute_mcse(x: np.ndarray, method: str = 'batch') -> float:
    """
    Compute Monte Carlo Standard Error (MCSE).
    
    MCSE estimates the standard error of the sample mean estimate.
    
    Args:
        x: MCMC samples
        method: 'batch' or 'spectral'
        
    Returns:
        MCSE estimate
    """
    n = len(x)
    
    if method == 'batch':
        # Batch means method
        batch_size = int(np.sqrt(n))
        n_batches = n // batch_size
        
        batch_means = []
        for i in range(n_batches):
            batch = x[i*batch_size:(i+1)*batch_size]
            batch_means.append(np.mean(batch))
        
        # Standard error of batch means
        se_batch = np.std(batch_means, ddof=1) / np.sqrt(n_batches)
        
        return se_batch
    else:
        # Spectral method using autocorrelations
        tau = integrated_autocorrelation_time(x)
        var_x = np.var(x, ddof=1)
        
        return np.sqrt(var_x * tau / n)


if __name__ == "__main__":
    # Test MCMC diagnostics
    print("Testing MCMC diagnostics...")
    
    np.random.seed(42)
    
    # Generate AR(1) process as mock MCMC chain
    n = 10000
    rho = 0.9  # High autocorrelation
    
    samples = np.zeros(n)
    samples[0] = np.random.normal()
    
    for i in range(1, n):
        samples[i] = rho * samples[i-1] + np.sqrt(1 - rho**2) * np.random.normal()
    
    # Compute diagnostics
    diag = diagnose_chain(samples)
    
    print(f"\nChain diagnostics:")
    print(f"  Number of samples: {diag['n_samples']}")
    print(f"  Mean: {diag['mean']:.4f}")
    print(f"  ESS: {diag['ess']:.1f}")
    print(f"  ESS/n: {diag['ess_per_sample']:.3f}")
    print(f"  Integrated ACF time: {diag['tau_int']:.1f}")
    
    print("\n✓ MCMC diagnostics tests completed")