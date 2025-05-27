"""
Test configuration and fixtures for lattice Gaussian MCMC project.

This module provides pytest fixtures and configuration for testing
the lattice Gaussian sampling algorithms and related utilities.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lattices.base import BaseLattice
from lattices.identity import IdentityLattice
from lattices.qary import QaryLattice
from samplers.klein import KleinSampler
from samplers.imhk import IMHKSampler
from lattices.reduction import LatticeReduction


@pytest.fixture(scope="session")
def test_seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp(prefix="lattice_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def simple_2d_basis():
    """Simple 2D lattice basis for testing."""
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)


@pytest.fixture
def random_2d_basis(test_seed):
    """Random 2D lattice basis."""
    np.random.seed(test_seed)
    # Generate a well-conditioned random basis
    A = np.random.randn(2, 2)
    U, _, Vt = np.linalg.svd(A)
    # Ensure condition number is reasonable
    singular_values = np.array([2.0, 1.0])
    return U @ np.diag(singular_values) @ Vt


@pytest.fixture
def pathological_basis():
    """Pathological basis with bad conditioning."""
    return np.array([[1.0, 0.0], [1e-10, 1.0]], dtype=float)


@pytest.fixture
def standard_3d_basis():
    """Standard 3D integer lattice basis."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0]
    ], dtype=float)


@pytest.fixture
def ntru_like_basis():
    """NTRU-like basis for cryptographic testing."""
    n = 8  # Small dimension for testing
    q = 17
    # Construct a simple NTRU-like basis
    h = np.random.randint(0, q, n)
    
    # Top half: [I | h]
    top = np.hstack([np.eye(n), h.reshape(-1, 1)])
    # Bottom half: [0 | q]
    bottom = np.hstack([np.zeros((1, n)), np.array([[q]])])
    
    return np.vstack([top, bottom]).astype(float)


@pytest.fixture
def identity_lattice_2d():
    """2D identity lattice instance."""
    return IdentityLattice(dimension=2)


@pytest.fixture
def identity_lattice_3d():
    """3D identity lattice instance."""
    return IdentityLattice(dimension=3)


@pytest.fixture
def qary_lattice_small():
    """Small q-ary lattice for testing."""
    # Create simple LWE-style matrix
    n, m, q = 4, 6, 17
    A = np.random.randint(0, q, (n, m))
    return QaryLattice.from_lwe_instance(A, q, dual=False)


@pytest.fixture
def klein_sampler_2d(identity_lattice_2d):
    """Klein sampler for 2D identity lattice."""
    return KleinSampler(identity_lattice_2d, sigma=1.0)


@pytest.fixture
def imhk_sampler_2d(identity_lattice_2d):
    """IMHK sampler for 2D identity lattice."""
    return IMHKSampler(identity_lattice_2d, sigma=1.0)


@pytest.fixture
def tolerance_config():
    """Standard tolerance configuration for numerical tests."""
    return {
        'rtol': 1e-10,          # Relative tolerance
        'atol': 1e-12,          # Absolute tolerance
        'statistical_rtol': 0.1,  # Statistical test tolerance (10%)
        'condition_threshold': 1e12,  # Matrix conditioning threshold
        'spectral_gap_tol': 1e-8     # Spectral gap computation tolerance
    }


@pytest.fixture
def statistical_config():
    """Configuration for statistical tests."""
    return {
        'n_samples': 10000,      # Number of samples for statistical tests
        'n_trials': 100,         # Number of trials for Monte Carlo tests
        'confidence_level': 0.95, # Confidence level for statistical tests
        'min_ess': 100,          # Minimum effective sample size
        'max_autocorr_lag': 100  # Maximum lag for autocorrelation
    }


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'max_time_simple': 1.0,     # Max time for simple operations (seconds)
        'max_time_sampling': 10.0,   # Max time for sampling operations
        'max_time_reduction': 30.0,  # Max time for reduction operations
        'memory_limit_mb': 100       # Memory limit in MB
    }


@pytest.fixture
def reference_data():
    """Reference data for regression testing."""
    return {
        'klein_2d_samples_mean': np.array([0.0, 0.0]),
        'klein_2d_samples_cov': np.eye(2),
        'spectral_gaps': {
            'identity_2d_sigma_1': 0.8,  # Approximate expected values
            'identity_3d_sigma_1': 0.7
        },
        'mixing_times': {
            'identity_2d_sigma_1': 10,   # Approximate expected values
            'identity_3d_sigma_1': 15
        }
    }


@pytest.fixture
def golden_files_config(temp_dir):
    """Configuration for golden file testing."""
    golden_dir = temp_dir / "golden"
    golden_dir.mkdir(exist_ok=True)
    
    return {
        'golden_dir': golden_dir,
        'figure_formats': ['png', 'pdf'],
        'data_formats': ['json', 'npz'],
        'tolerance': 1e-10
    }


class TestMatrixUtils:
    """Utility functions for matrix testing."""
    
    @staticmethod
    def is_basis_valid(basis: np.ndarray, tol: float = 1e-12) -> bool:
        """Check if a matrix forms a valid lattice basis."""
        if basis.ndim != 2:
            return False
        if basis.shape[0] > basis.shape[1]:
            return False
        # Check if rows are linearly independent
        rank = np.linalg.matrix_rank(basis, tol=tol)
        return rank == basis.shape[0]
    
    @staticmethod
    def condition_number(basis: np.ndarray) -> float:
        """Compute condition number of basis matrix."""
        return np.linalg.cond(basis)
    
    @staticmethod
    def gram_schmidt_orthogonality_defect(basis: np.ndarray) -> float:
        """Compute orthogonality defect of Gram-Schmidt basis."""
        Q, R = np.linalg.qr(basis.T)
        gram_schmidt_norms = np.abs(np.diag(R))
        basis_norms = np.linalg.norm(basis, axis=1)
        return np.prod(basis_norms) / np.prod(gram_schmidt_norms)


class TestStatUtils:
    """Utility functions for statistical testing."""
    
    @staticmethod
    def kolmogorov_smirnov_test(samples: np.ndarray, 
                               reference_cdf, 
                               alpha: float = 0.05) -> Tuple[bool, float]:
        """Perform Kolmogorov-Smirnov test against reference distribution."""
        from scipy import stats
        n = len(samples)
        sorted_samples = np.sort(samples)
        
        # Compute empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Compute reference CDF values
        reference_values = reference_cdf(sorted_samples)
        
        # Compute KS statistic
        ks_stat = np.max(np.abs(empirical_cdf - reference_values))
        
        # Critical value
        critical_value = stats.ksone.ppf(1 - alpha/2, n)
        
        return ks_stat < critical_value, ks_stat
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray, max_lag: int = None) -> float:
        """Compute effective sample size using autocorrelation."""
        if max_lag is None:
            max_lag = min(len(samples) // 4, 100)
        
        # Compute autocorrelation
        def autocorr(x, max_lag):
            n = len(x)
            x = x - np.mean(x)
            autocorr_func = np.correlate(x, x, mode='full')
            autocorr_func = autocorr_func[n-1:]
            autocorr_func = autocorr_func / autocorr_func[0]
            return autocorr_func[:max_lag+1]
        
        autocorr_values = autocorr(samples, max_lag)
        
        # Find first negative autocorrelation
        first_negative = np.where(autocorr_values < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(autocorr_values)
        
        # Compute integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr_values[1:cutoff])
        
        return len(samples) / (2 * tau_int) if tau_int > 0 else len(samples)


@pytest.fixture
def matrix_utils():
    """Matrix utility functions for testing."""
    return TestMatrixUtils()


@pytest.fixture  
def stat_utils():
    """Statistical utility functions for testing."""
    return TestStatUtils()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions/methods"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for module interactions"
    )
    config.addinivalue_line(
        "markers", "end_to_end: Full pipeline tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "statistical: Tests that verify statistical properties"
    )
    config.addinivalue_line(
        "markers", "numerical: Tests that verify numerical accuracy"
    )
    config.addinivalue_line(
        "markers", "edge_case: Tests for edge cases and error conditions"
    )
    config.addinivalue_line(
        "markers", "reproducibility: Tests for deterministic behavior"
    )


def pytest_runtest_setup(item):
    """Setup for each test item - ensure reproducible random state."""
    np.random.seed(42)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add unit marker to unit test files
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration test files  
        if "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ['reduction', 'sampling', 'mcmc']):
            item.add_marker(pytest.mark.slow)
            
        # Add statistical marker to statistical tests
        if any(keyword in item.name.lower() for keyword in ['statistical', 'distribution', 'convergence']):
            item.add_marker(pytest.mark.statistical)
            
        # Add numerical marker to numerical accuracy tests
        if any(keyword in item.name.lower() for keyword in ['accuracy', 'precision', 'numerical']):
            item.add_marker(pytest.mark.numerical)
            
        # Add edge_case marker to edge case tests
        if any(keyword in item.name.lower() for keyword in ['edge', 'pathological', 'degenerate']):
            item.add_marker(pytest.mark.edge_case)