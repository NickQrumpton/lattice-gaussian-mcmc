"""
LatticeReduction: Comprehensive lattice reduction algorithms and diagnostics.

Implements state-of-the-art reduction algorithms with quality diagnostics
tailored for discrete Gaussian sampling and cryptographic research.
"""

import numpy as np
import matplotlib.pyplot as plt
from sage.all import (
    Matrix, vector, RDF, RR, ZZ, QQ, sqrt, log, ln, exp, pi,
    identity_matrix, block_matrix, copy, prod,
    LLL, BKZ, GSO
)
import time
from typing import Tuple, List, Dict, Optional, Union, Any
import logging
import json
import os
from datetime import datetime


class LatticeReduction:
    """
    Comprehensive lattice reduction algorithms with diagnostics.
    
    Provides implementations and wrappers for LLL, BKZ, and specialized
    reductions optimized for discrete Gaussian sampling.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize lattice reduction module.
        
        Args:
            log_dir: Directory for logging reduction progress
        """
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Setup logging
        self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'reductions_performed': 0,
            'total_time': 0.0,
            'quality_improvements': []
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
        
        if self.log_dir:
            # Add file handler
            log_file = os.path.join(self.log_dir, 
                                   f'reduction_{datetime.now():%Y%m%d_%H%M%S}.log')
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(fh)
            
    # ========== 1. LLL Reduction ==========
    
    def lll_reduce(self, basis: Matrix, delta: float = 0.75,
                   track_progress: bool = False) -> Tuple[Matrix, Matrix, Dict]:
        """
        LLL reduction with enhanced diagnostics.
        
        Args:
            basis: Input basis matrix
            delta: LLL parameter (0.25 < delta < 1)
            track_progress: Track quality measures during reduction
            
        Returns:
            tuple: (reduced_basis, transformation_matrix, diagnostics)
        """
        self.logger.info(f"Starting LLL reduction with ´={delta}")
        start_time = time.time()
        
        # Initial quality metrics
        initial_quality = self.basis_quality_profile(basis)
        
        # Convert to appropriate type
        B = Matrix(QQ, basis)
        n = B.nrows()
        
        # Initialize transformation matrix
        U = identity_matrix(QQ, n)
        
        if track_progress:
            progress_log = []
            
        # LLL reduction with tracking
        B_reduced = copy(B)
        
        # Use SageMath's LLL with custom tracking
        if not track_progress:
            # Simple case - use built-in LLL
            B_reduced = B.LLL(delta=delta)
            U = B.solve_left(B_reduced)
        else:
            # Manual LLL for tracking
            B_reduced, U, progress_log = self._lll_with_tracking(B, delta)
            
        # Final quality metrics
        final_quality = self.basis_quality_profile(B_reduced)
        
        # Compute diagnostics
        reduction_time = time.time() - start_time
        diagnostics = {
            'time': reduction_time,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'improvement_factor': initial_quality['hermite_factor'] / final_quality['hermite_factor'],
            'delta': delta
        }
        
        if track_progress:
            diagnostics['progress'] = progress_log
            
        # Update statistics
        self.stats['reductions_performed'] += 1
        self.stats['total_time'] += reduction_time
        self.stats['quality_improvements'].append(diagnostics['improvement_factor'])
        
        self.logger.info(f"LLL completed in {reduction_time:.2f}s, "
                        f"improvement factor: {diagnostics['improvement_factor']:.2f}")
        
        return Matrix(RDF, B_reduced), Matrix(RDF, U), diagnostics
    
    def _lll_with_tracking(self, B: Matrix, delta: float) -> Tuple[Matrix, Matrix, List]:
        """
        LLL with step-by-step tracking.
        
        Internal method for detailed progress monitoring.
        """
        n = B.nrows()
        B_work = copy(B)
        U = identity_matrix(QQ, n)
        progress = []
        
        # Gram-Schmidt orthogonalization
        gs = GSO(B_work)
        gs.update_gso()
        
        k = 1
        while k < n:
            # Size reduce
            for j in range(k-1, -1, -1):
                mu_kj = gs.mu[k, j]
                if abs(mu_kj) > 0.5:
                    q = round(mu_kj)
                    B_work[k] = B_work[k] - q * B_work[j]
                    U[k] = U[k] - q * U[j]
                    gs.update_gso()
                    
            # Lovász condition
            if k > 0:
                lhs = gs.r[k, k]
                rhs = (delta - gs.mu[k, k-1]**2) * gs.r[k-1, k-1]
                
                if lhs < rhs:
                    # Swap
                    B_work.swap_rows(k, k-1)
                    U.swap_rows(k, k-1)
                    gs.update_gso()
                    k = max(k-1, 1)
                    
                    # Log progress
                    if len(progress) % 10 == 0:  # Log every 10 swaps
                        quality = self._quick_quality(B_work)
                        progress.append({
                            'step': len(progress),
                            'k': k,
                            'hermite_factor': quality
                        })
                else:
                    k += 1
            else:
                k += 1
                
        return B_work, U, progress
    
    def lll_with_removals(self, basis: Matrix, target_dim: Optional[int] = None,
                         removal_threshold: float = 0.99) -> Tuple[Matrix, List[int]]:
        """
        LLL with removal strategy for approximate SVP.
        
        Args:
            basis: Input basis
            target_dim: Target dimension after removal
            removal_threshold: Threshold for vector removal
            
        Returns:
            tuple: (reduced_basis, removed_indices)
        """
        self.logger.info("Starting LLL with removals")
        
        B = Matrix(QQ, basis)
        n = B.nrows()
        
        if target_dim is None:
            target_dim = max(1, n // 2)
            
        # Initial LLL
        B_reduced, _, _ = self.lll_reduce(B)
        
        # Compute GS norms
        gs = GSO(B_reduced)
        gs.update_gso()
        gs_norms = [float(gs.r[i, i]) for i in range(n)]
        
        # Identify vectors to remove
        removed = []
        threshold = removal_threshold * max(gs_norms)
        
        for i in range(n-1, -1, -1):
            if len(removed) >= n - target_dim:
                break
            if gs_norms[i] > threshold:
                removed.append(i)
                
        # Create reduced basis
        keep_indices = [i for i in range(n) if i not in removed]
        B_final = B_reduced[keep_indices, :]
        
        self.logger.info(f"Removed {len(removed)} vectors, "
                        f"final dimension: {len(keep_indices)}")
        
        return B_final, removed
    
    # ========== 2. BKZ Reduction ==========
    
    def bkz_reduce(self, basis: Matrix, block_size: int,
                   strategy: str = 'default',
                   progressive: bool = False) -> Tuple[Matrix, Matrix, Dict]:
        """
        BKZ reduction with various strategies.
        
        Args:
            basis: Input basis
            block_size: BKZ block size
            strategy: 'default', 'bkz2', or custom
            progressive: Use progressive block sizes
            
        Returns:
            tuple: (reduced_basis, transformation, diagnostics)
        """
        self.logger.info(f"Starting BKZ reduction with block_size={block_size}")
        start_time = time.time()
        
        # Initial quality
        initial_quality = self.basis_quality_profile(basis)
        
        B = Matrix(QQ, basis)
        n = B.nrows()
        
        # Progressive strategy
        if progressive:
            block_sizes = self._progressive_block_sizes(n, block_size)
        else:
            block_sizes = [block_size]
            
        # Apply BKZ
        B_reduced = copy(B)
        for beta in block_sizes:
            self.logger.info(f"  BKZ with block size {beta}")
            
            if strategy == 'default':
                # Use SageMath's BKZ
                B_reduced = B_reduced.BKZ(block_size=beta)
            else:
                # Custom strategies (placeholder)
                B_reduced = self._custom_bkz(B_reduced, beta, strategy)
                
        # Compute transformation
        U = B.solve_left(B_reduced)
        
        # Final quality
        final_quality = self.basis_quality_profile(B_reduced)
        
        # Diagnostics
        reduction_time = time.time() - start_time
        diagnostics = {
            'time': reduction_time,
            'block_size': block_size,
            'block_sizes_used': block_sizes,
            'strategy': strategy,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'improvement_factor': initial_quality['hermite_factor'] / final_quality['hermite_factor']
        }
        
        self.logger.info(f"BKZ completed in {reduction_time:.2f}s")
        
        return Matrix(RDF, B_reduced), Matrix(RDF, U), diagnostics
    
    def _progressive_block_sizes(self, n: int, target: int) -> List[int]:
        """Generate progressive block sizes."""
        sizes = []
        beta = min(20, n)
        
        while beta < target:
            sizes.append(beta)
            beta = min(beta + 10, target)
            
        sizes.append(target)
        return sizes
    
    def _custom_bkz(self, basis: Matrix, block_size: int, 
                    strategy: str) -> Matrix:
        """Placeholder for custom BKZ strategies."""
        # For now, fall back to default
        return basis.BKZ(block_size=block_size)
    
    # ========== 3. Basis Quality Analytics ==========
    
    def hermite_factor(self, basis: Matrix) -> float:
        """
        Compute Hermite factor of the basis.
        
        ³ = ||b_1|| / (det(›)^(1/n))
        
        Args:
            basis: Lattice basis
            
        Returns:
            float: Hermite factor
        """
        B = Matrix(RDF, basis)
        n = B.nrows()
        
        # First vector norm
        b1_norm = float(B[0].norm())
        
        # Determinant
        det = abs(B.det())
        
        # Hermite factor
        gamma = b1_norm / (det**(1/n))
        
        return gamma
    
    def orthogonality_defect(self, basis: Matrix) -> float:
        """
        Compute orthogonality defect.
        
        OD = ||b_i|| / det(›)
        
        Args:
            basis: Lattice basis
            
        Returns:
            float: Orthogonality defect
        """
        B = Matrix(RDF, basis)
        
        # Product of norms
        norm_product = prod(B[i].norm() for i in range(B.nrows()))
        
        # Determinant
        det = abs(B.det())
        
        # Orthogonality defect
        od = float(norm_product / det)
        
        return od
    
    def basis_quality_profile(self, basis: Matrix) -> Dict[str, Any]:
        """
        Comprehensive basis quality analysis.
        
        Args:
            basis: Lattice basis
            
        Returns:
            dict: Quality metrics including GS norms
        """
        B = Matrix(QQ, basis)
        n = B.nrows()
        
        # Gram-Schmidt
        gs = GSO(B)
        gs.update_gso()
        
        # Extract GS norms
        gs_norms = [float(sqrt(gs.r[i, i])) for i in range(n)]
        
        # Quality metrics
        profile = {
            'dimension': n,
            'hermite_factor': self.hermite_factor(B),
            'orthogonality_defect': self.orthogonality_defect(B),
            'gs_norms': gs_norms,
            'max_gs_norm': max(gs_norms),
            'min_gs_norm': min(gs_norms),
            'gs_ratio': max(gs_norms) / min(gs_norms),
            'log_potential': sum(log(norm) for norm in gs_norms)
        }
        
        return profile
    
    # ========== 4. Sampling-Optimized Reductions ==========
    
    def sampling_reduce(self, basis: Matrix, target_sigma: float,
                       max_iterations: int = 10) -> Tuple[Matrix, Dict]:
        """
        Optimize basis for discrete Gaussian sampling.
        
        Minimizes max||b*_i|| while maintaining stability for sampling
        at the target Ã.
        
        Args:
            basis: Input basis
            target_sigma: Target standard deviation for sampling
            max_iterations: Maximum optimization iterations
            
        Returns:
            tuple: (optimized_basis, optimization_info)
        """
        self.logger.info(f"Optimizing basis for sampling with Ã={target_sigma}")
        
        B = Matrix(QQ, basis)
        n = B.nrows()
        best_basis = copy(B)
        best_max_norm = float('inf')
        
        optimization_log = []
        
        for iteration in range(max_iterations):
            # Try different reduction strategies
            candidates = []
            
            # LLL with varying delta
            for delta in [0.75, 0.85, 0.95, 0.99]:
                B_lll, _, _ = self.lll_reduce(B, delta=delta)
                candidates.append(('LLL', delta, B_lll))
                
            # BKZ with small block sizes
            for beta in [20, 30, 40]:
                if beta < n:
                    B_bkz, _, _ = self.bkz_reduce(B, block_size=beta)
                    candidates.append(('BKZ', beta, B_bkz))
                    
            # Evaluate candidates
            for method, param, B_cand in candidates:
                profile = self.basis_quality_profile(B_cand)
                max_norm = profile['max_gs_norm']
                
                # Check if suitable for sampling
                if max_norm < target_sigma * sqrt(n):
                    if max_norm < best_max_norm:
                        best_basis = B_cand
                        best_max_norm = max_norm
                        
                        optimization_log.append({
                            'iteration': iteration,
                            'method': method,
                            'parameter': param,
                            'max_gs_norm': max_norm,
                            'improvement': best_max_norm / max_norm
                        })
                        
            # Local improvements
            B = self.gram_schmidt_improve(best_basis)
            
            # Check convergence
            if len(optimization_log) > 1:
                if optimization_log[-1]['improvement'] < 1.01:
                    break
                    
        # Final profile
        final_profile = self.basis_quality_profile(best_basis)
        
        optimization_info = {
            'iterations': len(optimization_log),
            'final_max_norm': best_max_norm,
            'suitable_for_sigma': best_max_norm < target_sigma * sqrt(n),
            'optimization_log': optimization_log,
            'final_profile': final_profile
        }
        
        self.logger.info(f"Optimization complete: max||b*_i|| = {best_max_norm:.2f}")
        
        return Matrix(RDF, best_basis), optimization_info
    
    def gram_schmidt_improve(self, basis: Matrix, 
                           max_swaps: int = 100) -> Matrix:
        """
        Local Gram-Schmidt improvements.
        
        Args:
            basis: Input basis
            max_swaps: Maximum local operations
            
        Returns:
            Matrix: Improved basis
        """
        B = Matrix(QQ, basis)
        n = B.nrows()
        improved = False
        swaps = 0
        
        # Try local swaps to improve GS profile
        for _ in range(max_swaps):
            gs = GSO(B)
            gs.update_gso()
            
            # Find adjacent vectors to swap
            best_swap = None
            best_improvement = 1.0
            
            for i in range(n-1):
                # Simulate swap
                if gs.r[i, i] > gs.r[i+1, i+1]:
                    ratio = gs.r[i, i] / gs.r[i+1, i+1]
                    if ratio > best_improvement:
                        best_swap = i
                        best_improvement = ratio
                        
            if best_swap is not None and best_improvement > 1.01:
                B.swap_rows(best_swap, best_swap + 1)
                swaps += 1
                improved = True
            else:
                break
                
        if improved:
            self.logger.debug(f"Gram-Schmidt improvement: {swaps} swaps")
            
        return B
    
    # ========== 5. Diagnostic and Analysis Tools ==========
    
    def plot_gram_schmidt_profile(self, bases: Union[Matrix, List[Matrix]], 
                                labels: Optional[List[str]] = None,
                                save_path: Optional[str] = None,
                                log_scale: bool = True):
        """
        Visualize Gram-Schmidt profiles.
        
        Args:
            bases: Single basis or list of bases to compare
            labels: Labels for each basis
            save_path: Path to save figure
            log_scale: Use log scale for y-axis
        """
        if isinstance(bases, Matrix):
            bases = [bases]
            
        if labels is None:
            labels = [f'Basis {i+1}' for i in range(len(bases))]
            
        plt.figure(figsize=(10, 6))
        
        for basis, label in zip(bases, labels):
            profile = self.basis_quality_profile(basis)
            gs_norms = profile['gs_norms']
            indices = list(range(len(gs_norms)))
            
            plt.plot(indices, gs_norms, 'o-', label=label, linewidth=2)
            
        plt.xlabel('Basis Vector Index', fontsize=12)
        plt.ylabel('Gram-Schmidt Norm $\|b_i^*\|$', fontsize=12)
        plt.title('Gram-Schmidt Profile Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.yscale('log')
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def estimate_reduction_cost(self, basis: Matrix, block_size: int,
                               model: str = 'bkz2') -> Dict[str, float]:
        """
        Estimate reduction cost.
        
        Args:
            basis: Input basis
            block_size: BKZ block size
            model: Cost model ('bkz2', 'classical', 'quantum')
            
        Returns:
            dict: Cost estimates
        """
        n = basis.nrows()
        
        # Simplified cost models
        if model == 'bkz2':
            # BKZ 2.0 cost model
            enum_cost = 2**(0.292 * block_size)  # Enumeration
            tours = n / block_size
            total_ops = tours * enum_cost
            
        elif model == 'classical':
            # Classical sieving
            sieve_cost = 2**(0.292 * block_size + 16.4)
            total_ops = sieve_cost
            
        else:  # quantum
            # Quantum speedup
            quantum_cost = 2**(0.265 * block_size)
            total_ops = quantum_cost
            
        # Time estimates (very rough)
        ops_per_second = 1e9  # 1 GHz estimate
        time_seconds = total_ops / ops_per_second
        
        return {
            'model': model,
            'block_size': block_size,
            'dimension': n,
            'operations': total_ops,
            'log2_operations': float(log(total_ops, 2)),
            'estimated_seconds': time_seconds,
            'estimated_hours': time_seconds / 3600
        }
    
    def compare_bases(self, basis1: Matrix, basis2: Matrix,
                     labels: Tuple[str, str] = ('Before', 'After')) -> Dict:
        """
        Compare two bases.
        
        Args:
            basis1: First basis
            basis2: Second basis
            labels: Labels for the bases
            
        Returns:
            dict: Comparison results
        """
        profile1 = self.basis_quality_profile(basis1)
        profile2 = self.basis_quality_profile(basis2)
        
        comparison = {
            'labels': labels,
            'hermite_factor': {
                labels[0]: profile1['hermite_factor'],
                labels[1]: profile2['hermite_factor'],
                'improvement': profile1['hermite_factor'] / profile2['hermite_factor']
            },
            'orthogonality_defect': {
                labels[0]: profile1['orthogonality_defect'],
                labels[1]: profile2['orthogonality_defect'],
                'improvement': profile1['orthogonality_defect'] / profile2['orthogonality_defect']
            },
            'max_gs_norm': {
                labels[0]: profile1['max_gs_norm'],
                labels[1]: profile2['max_gs_norm'],
                'improvement': profile1['max_gs_norm'] / profile2['max_gs_norm']
            },
            'gs_ratio': {
                labels[0]: profile1['gs_ratio'],
                labels[1]: profile2['gs_ratio'],
                'improvement': profile1['gs_ratio'] / profile2['gs_ratio']
            }
        }
        
        # Generate comparison plot
        self.plot_gram_schmidt_profile([basis1, basis2], labels=labels)
        
        return comparison
    
    def save_reduction_report(self, results: Dict, filename: str):
        """
        Save reduction results to file.
        
        Args:
            results: Reduction results
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            else:
                return obj
                
        results_clean = convert_arrays(results)
        
        with open(filename, 'w') as f:
            json.dump(results_clean, f, indent=2)
            
        self.logger.info(f"Reduction report saved to {filename}")
        

# Utility functions

def optimal_delta_for_sampling(n: int, target_sigma: float) -> float:
    """
    Heuristic for optimal LLL ´ parameter for sampling.
    
    Args:
        n: Lattice dimension
        target_sigma: Target sampling parameter
        
    Returns:
        float: Recommended ´ value
    """
    # Higher ´ gives better reduction but is slower
    # For sampling, we want good reduction
    if n < 50:
        return 0.99
    elif n < 100:
        return 0.95
    elif n < 200:
        return 0.85
    else:
        return 0.75
        

def reduction_strategy_for_lattice(lattice_type: str, n: int) -> Dict[str, Any]:
    """
    Recommend reduction strategy based on lattice type.
    
    Args:
        lattice_type: Type of lattice ('random', 'qary', 'ntru', etc.)
        n: Dimension
        
    Returns:
        dict: Recommended parameters
    """
    strategies = {
        'random': {
            'algorithm': 'LLL',
            'delta': 0.99,
            'use_bkz': n < 100,
            'bkz_block': min(40, n // 2)
        },
        'qary': {
            'algorithm': 'BKZ',
            'delta': 0.95,
            'use_bkz': True,
            'bkz_block': min(50, n // 2),
            'progressive': True
        },
        'ntru': {
            'algorithm': 'BKZ',
            'delta': 0.99,
            'use_bkz': True,
            'bkz_block': min(60, n // 3),
            'progressive': True
        },
        'ideal': {
            'algorithm': 'LLL',
            'delta': 0.85,
            'use_bkz': n < 200,
            'bkz_block': min(30, n // 4)
        }
    }
    
    return strategies.get(lattice_type, strategies['random'])


# Example usage
if __name__ == '__main__':
    from sage.all import random_matrix
    
    # Create test lattice
    n = 40
    B = random_matrix(ZZ, n, n, x=-100, y=100)
    B = B.T * B  # Make it positive definite
    
    # Initialize reducer
    reducer = LatticeReduction()
    
    # Test LLL
    print("Testing LLL reduction...")
    B_lll, U_lll, info_lll = reducer.lll_reduce(B, delta=0.99, track_progress=True)
    print(f"LLL improvement factor: {info_lll['improvement_factor']:.2f}")
    
    # Test BKZ
    print("\nTesting BKZ reduction...")
    B_bkz, U_bkz, info_bkz = reducer.bkz_reduce(B, block_size=20, progressive=True)
    print(f"BKZ improvement factor: {info_bkz['improvement_factor']:.2f}")
    
    # Test sampling optimization
    print("\nTesting sampling optimization...")
    B_opt, opt_info = reducer.sampling_reduce(B, target_sigma=100.0)
    print(f"Optimized max GS norm: {opt_info['final_max_norm']:.2f}")
    
    # Compare bases
    print("\nComparing bases...")
    comparison = reducer.compare_bases(B, B_opt)
    
    # Cost estimation
    print("\nCost estimation for BKZ-60:")
    cost = reducer.estimate_reduction_cost(B, block_size=60)
    print(f"Estimated operations: 2^{cost['log2_operations']:.1f}")