#!/usr/bin/env python3
"""
Runner script for Klein sampler validation experiments.

This script executes all validation experiments with carefully chosen
parameters to demonstrate the correctness of the Klein sampler implementation.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.klein_validation_suite import KleinValidationExperiments


def run_validation_suite():
    """Run complete validation suite with research-quality parameters."""
    
    # Create output directory
    output_dir = "results/klein_validation"
    
    print("Klein Sampler Validation Suite")
    print("==============================")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize validator
    validator = KleinValidationExperiments(output_dir=output_dir)
    
    # Define experiment parameters
    
    # Experiment 1: 1D validation with different sigma values
    print("Running 1D validation experiments...")
    
    # Standard deviation σ = 10 (moderate width)
    validator.experiment_1d_validation(
        n_samples=100000,
        sigma=10.0,
        center=0.0
    )
    
    # Additional test with σ = 2 (narrow distribution)
    print("\nRunning 1D validation with σ = 2...")
    validator.experiment_1d_validation(
        n_samples=100000,
        sigma=2.0,
        center=5.0  # Non-zero center
    )
    
    # Experiment 2: 2D Klein validation
    print("\nRunning 2D Klein sampler validation...")
    
    # Well-conditioned basis
    basis_1 = np.array([[4.0, 1.0], [1.0, 3.0]])
    validator.experiment_2d_klein_validation(
        n_samples=50000,
        basis=basis_1,
        sigma=2.0,
        center=np.array([0.0, 0.0])
    )
    
    # Skewed basis (more challenging)
    print("\nRunning 2D validation with skewed basis...")
    basis_2 = np.array([[5.0, 2.0], [1.0, 4.0]])
    validator.experiment_2d_klein_validation(
        n_samples=50000,
        basis=basis_2,
        sigma=3.0,
        center=np.array([1.0, -0.5])
    )
    
    # Experiment 3: Acceptance consistency
    print("\nRunning acceptance consistency experiments...")
    
    # Run with different sigma values to see acceptance rate variation
    for sigma in [1.0, 2.0, 5.0]:
        print(f"\nTesting acceptance with σ = {sigma}")
        validator.experiment_acceptance_consistency(
            n_steps=20000,
            basis=basis_1,
            sigma=sigma
        )
    
    # Experiment 4: Mixing time analysis
    print("\nRunning mixing time analysis...")
    
    # Standard parameters
    validator.experiment_mixing_time(
        n_steps=5000,
        basis=basis_1,
        sigma=2.0
    )
    
    # Longer chain for better statistics
    print("\nRunning extended mixing analysis...")
    validator.experiment_mixing_time(
        n_steps=10000,
        basis=basis_1,
        sigma=3.0
    )
    
    # Generate final report
    print("\nGenerating comprehensive report...")
    report = validator.generate_report()
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUITE COMPLETED")
    print("="*60)
    print("\nKey Results Summary:")
    
    if 'experiment_1d' in validator.results:
        res = validator.results['experiment_1d']
        print(f"\n1D Sampler:")
        print(f"  - TV Distance: {res['tv_distance']:.6f} (target: < 0.02)")
        print(f"  - Status: {'✓ PASS' if res['tv_distance'] < 0.02 else '✗ FAIL'}")
    
    if 'experiment_2d' in validator.results:
        res = validator.results['experiment_2d']
        print(f"\n2D Klein Sampler:")
        print(f"  - TV Distance: {res['tv_distance']:.6f} (target: < 0.02)")
        print(f"  - KL Divergence: {res['kl_divergence']:.6f} (target: < 0.05)")
        print(f"  - Status: {'✓ PASS' if res['tv_distance'] < 0.02 and res['kl_divergence'] < 0.05 else '✗ FAIL'}")
    
    if 'experiment_acceptance' in validator.results:
        res = validator.results['experiment_acceptance']
        print(f"\nAcceptance Consistency:")
        print(f"  - Acceptance Rate: {res['overall_acceptance']:.3f}")
        print(f"  - Status: {'✓ PASS' if res['overall_acceptance'] > 0.3 else '✗ FAIL'}")
    
    if 'experiment_mixing' in validator.results:
        res = validator.results['experiment_mixing']
        print(f"\nMixing Properties:")
        print(f"  - ESS/n ratio (x): {res['ess_x']/res['n_steps']:.3f} (target: > 0.1)")
        print(f"  - ESS/n ratio (y): {res['ess_y']/res['n_steps']:.3f} (target: > 0.1)")
        print(f"  - Status: {'✓ PASS' if min(res['ess_x'], res['ess_y'])/res['n_steps'] > 0.1 else '✗ FAIL'}")
    
    print(f"\nFull report saved to: {Path(output_dir) / 'validation_report.txt'}")
    print(f"Figures saved to: {output_dir}/")
    print("\n" + "="*60)


def run_quick_validation():
    """Run a quick validation with reduced sample sizes for testing."""
    
    print("Quick Validation Mode (reduced samples)")
    print("="*40)
    
    validator = KleinValidationExperiments(output_dir="results/klein_validation_quick")
    
    # Quick 1D test
    validator.experiment_1d_validation(n_samples=10000, sigma=5.0)
    
    # Quick 2D test
    basis = np.array([[4.0, 1.0], [1.0, 3.0]])
    validator.experiment_2d_klein_validation(n_samples=5000, basis=basis, sigma=2.0)
    
    # Quick acceptance test
    validator.experiment_acceptance_consistency(n_steps=2000, basis=basis, sigma=2.0)
    
    # Quick mixing test
    validator.experiment_mixing_time(n_steps=1000, basis=basis, sigma=2.0)
    
    # Generate report
    validator.generate_report()
    
    print("\nQuick validation completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Klein sampler validation experiments")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick validation with reduced samples")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_validation()
    else:
        run_validation_suite()