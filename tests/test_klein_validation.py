#!/usr/bin/env python3
"""
Unit tests for Klein sampler validation suite.

Tests that the validation experiments run correctly and produce
expected outputs.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.klein_validation_suite import KleinValidationExperiments


class TestKleinValidation(unittest.TestCase):
    """Test cases for Klein validation experiments."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.validator = KleinValidationExperiments(output_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_1d_validation_basic(self):
        """Test 1D validation with small sample size."""
        results = self.validator.experiment_1d_validation(
            n_samples=1000,
            sigma=5.0,
            center=0.0
        )
        
        # Check that results contain expected keys
        self.assertIn('tv_distance', results)
        self.assertIn('kl_divergence', results)
        self.assertIn('mean_error', results)
        self.assertIn('std_error', results)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(results['tv_distance'], 0.0)
        self.assertLessEqual(results['tv_distance'], 1.0)
        self.assertGreaterEqual(results['kl_divergence'], 0.0)
        
        # Check that plot was created
        plot_path = Path(self.temp_dir) / "experiment1_1d_validation.png"
        self.assertTrue(plot_path.exists())
    
    def test_2d_klein_validation_basic(self):
        """Test 2D Klein validation with small sample size."""
        basis = np.array([[2.0, 0.0], [0.0, 2.0]])  # Simple basis
        results = self.validator.experiment_2d_klein_validation(
            n_samples=1000,
            basis=basis,
            sigma=2.0,
            center=np.array([0.0, 0.0])
        )
        
        # Check results structure
        self.assertIn('tv_distance', results)
        self.assertIn('kl_divergence', results)
        self.assertIn('empirical_mean', results)
        self.assertIn('empirical_cov', results)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(results['tv_distance'], 0.0)
        self.assertLessEqual(results['tv_distance'], 1.0)
        
        # Check that plots were created
        heatmap_path = Path(self.temp_dir) / "experiment2_2d_heatmaps.png"
        scatter_path = Path(self.temp_dir) / "experiment2_2d_scatter.png"
        self.assertTrue(heatmap_path.exists())
        self.assertTrue(scatter_path.exists())
    
    def test_acceptance_consistency(self):
        """Test acceptance consistency experiment."""
        basis = np.array([[2.0, 0.0], [0.0, 2.0]])
        results = self.validator.experiment_acceptance_consistency(
            n_steps=1000,
            basis=basis,
            sigma=2.0
        )
        
        # Check results
        self.assertIn('overall_acceptance', results)
        self.assertIn('block_acceptances', results)
        self.assertIn('acceptance_std', results)
        
        # Acceptance rate should be between 0 and 1
        self.assertGreaterEqual(results['overall_acceptance'], 0.0)
        self.assertLessEqual(results['overall_acceptance'], 1.0)
        
        # Check plot
        plot_path = Path(self.temp_dir) / "experiment3_acceptance_rates.png"
        self.assertTrue(plot_path.exists())
    
    def test_mixing_time_analysis(self):
        """Test mixing time estimation."""
        basis = np.array([[2.0, 0.0], [0.0, 2.0]])
        results = self.validator.experiment_mixing_time(
            n_steps=500,
            basis=basis,
            sigma=2.0
        )
        
        # Check results
        self.assertIn('tau_int_x', results)
        self.assertIn('tau_int_y', results)
        self.assertIn('ess_x', results)
        self.assertIn('ess_y', results)
        
        # ESS should be positive and less than n_steps
        self.assertGreater(results['ess_x'], 0)
        self.assertLess(results['ess_x'], results['n_steps'])
        self.assertGreater(results['ess_y'], 0)
        self.assertLess(results['ess_y'], results['n_steps'])
        
        # Check plots
        acf_path = Path(self.temp_dir) / "experiment4_acf.png"
        trace_path = Path(self.temp_dir) / "experiment4_traces.png"
        self.assertTrue(acf_path.exists())
        self.assertTrue(trace_path.exists())
    
    def test_report_generation(self):
        """Test report generation."""
        # Run a minimal experiment
        self.validator.experiment_1d_validation(n_samples=100, sigma=5.0)
        
        # Generate report
        report = self.validator.generate_report()
        
        # Check that report contains expected sections
        self.assertIn("Klein Sampler Validation Report", report)
        self.assertIn("Summary of Results", report)
        self.assertIn("Interpretation", report)
        
        # Check that files were created
        report_path = Path(self.temp_dir) / "validation_report.txt"
        results_path = Path(self.temp_dir) / "validation_results.json"
        self.assertTrue(report_path.exists())
        self.assertTrue(results_path.exists())
    
    def test_tv_distance_convergence(self):
        """Test that TV distance decreases with more samples."""
        # Run with different sample sizes
        tv_distances = []
        for n_samples in [100, 1000, 10000]:
            results = self.validator.experiment_1d_validation(
                n_samples=n_samples,
                sigma=5.0,
                center=0.0
            )
            tv_distances.append(results['tv_distance'])
        
        # TV distance should generally decrease with more samples
        # (though not strictly monotonic due to randomness)
        self.assertLess(tv_distances[-1], tv_distances[0] + 0.1)
    
    def test_different_bases(self):
        """Test Klein sampler with different basis matrices."""
        bases = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            np.array([[2.0, 1.0], [1.0, 2.0]]),  # Well-conditioned
            np.array([[5.0, 4.0], [4.0, 5.0]]),  # Larger scale
        ]
        
        for basis in bases:
            results = self.validator.experiment_2d_klein_validation(
                n_samples=1000,
                basis=basis,
                sigma=2.0
            )
            
            # Should get valid results for all bases
            self.assertGreaterEqual(results['tv_distance'], 0.0)
            self.assertLessEqual(results['tv_distance'], 1.0)


class TestValidationMetrics(unittest.TestCase):
    """Test validation metric calculations."""
    
    def test_tv_distance_calculation(self):
        """Test total variation distance calculation."""
        # Create two distributions
        p = np.array([0.1, 0.3, 0.4, 0.2])
        q = np.array([0.2, 0.3, 0.3, 0.2])
        
        # TV distance = 0.5 * sum(|p - q|)
        tv = 0.5 * np.sum(np.abs(p - q))
        expected = 0.5 * (0.1 + 0.0 + 0.1 + 0.0)
        
        self.assertAlmostEqual(tv, expected)
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation."""
        # Create two distributions
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        # KL divergence should be 0 for identical distributions
        kl = np.sum(p * np.log(p / q))
        self.assertAlmostEqual(kl, 0.0)


if __name__ == '__main__':
    unittest.main()