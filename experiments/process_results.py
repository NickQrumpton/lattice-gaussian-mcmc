#!/usr/bin/env python3
"""
Process and display experiment results.
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd

# Check what files were created
sample_files = sorted(Path('results/samples').glob('*.npz'))
print(f"Found {len(sample_files)} sample files:")
for f in sample_files:
    print(f"  - {f.name}")

# Load and display identity lattice results
print("\n" + "="*60)
print("IDENTITY LATTICE RESULTS")
print("="*60)

# Manually recreate results from the output
identity_data = [
    {'lattice': 'identity', 'n': 16, 'regime': 'hard', 'sigma': 2.0, 'rate': 29705.8, 'mean_norm': 7.91, 'acf_1': -0.015},
    {'lattice': 'identity', 'n': 16, 'regime': 'near', 'sigma': 4.0, 'rate': 32926.0, 'mean_norm': 15.73, 'acf_1': 0.006},
    {'lattice': 'identity', 'n': 16, 'regime': 'smooth', 'sigma': 8.0, 'rate': 32704.4, 'mean_norm': 31.56, 'acf_1': 0.020},
    {'lattice': 'identity', 'n': 64, 'regime': 'hard', 'sigma': 4.0, 'rate': 11365.5, 'mean_norm': 31.84, 'acf_1': -0.025},
    {'lattice': 'identity', 'n': 64, 'regime': 'near', 'sigma': 8.0, 'rate': 11339.9, 'mean_norm': 63.78, 'acf_1': 0.068},
    {'lattice': 'identity', 'n': 64, 'regime': 'smooth', 'sigma': 16.0, 'rate': 10744.0, 'mean_norm': 127.45, 'acf_1': -0.017},
    {'lattice': 'identity', 'n': 256, 'regime': 'hard', 'sigma': 8.0, 'rate': 3029.3, 'mean_norm': 126.81, 'acf_1': 0.013},
    {'lattice': 'identity', 'n': 256, 'regime': 'near', 'sigma': 16.0, 'rate': 3150.3, 'mean_norm': 256.34, 'acf_1': -0.036},
    {'lattice': 'identity', 'n': 256, 'regime': 'smooth', 'sigma': 32.0, 'rate': 3091.9, 'mean_norm': 512.07, 'acf_1': -0.022},
]

# Convert to DataFrame for nice display
df_identity = pd.DataFrame(identity_data)
print("\nIdentity Lattice Performance:")
print(df_identity.to_string(index=False))

# Performance summary by dimension
print("\n" + "-"*40)
print("Performance Summary by Dimension:")
print("-"*40)
summary = df_identity.groupby('n')['rate'].agg(['mean', 'min', 'max'])
print(summary)

# Check sample properties
print("\n" + "-"*40)
print("Sample Quality Check:")
print("-"*40)

for _, row in df_identity.iterrows():
    n = row['n']
    expected_norm = row['sigma'] * np.sqrt(n)
    actual_norm = row['mean_norm']
    error = abs(actual_norm - expected_norm) / expected_norm * 100
    
    print(f"n={n:3d}, σ={row['sigma']:4.1f}: "
          f"Expected norm={expected_norm:6.2f}, "
          f"Actual={actual_norm:6.2f}, "
          f"Error={error:4.1f}%")

# Load actual sample data and compute more diagnostics
print("\n" + "="*60)
print("DETAILED SAMPLE ANALYSIS")
print("="*60)

for f in sample_files[:3]:  # Analyze first 3 files
    print(f"\nFile: {f.name}")
    data = np.load(f)
    samples = data['samples']
    norms = data['norms']
    
    print(f"  Shape: {samples.shape}")
    print(f"  Norm statistics:")
    print(f"    Mean: {np.mean(norms):.2f}")
    print(f"    Std:  {np.std(norms):.2f}")
    print(f"    Min:  {np.min(norms):.2f}")
    print(f"    Max:  {np.max(norms):.2f}")
    
    # Check discreteness
    is_discrete = np.all(samples == samples.astype(int))
    print(f"  All integer values: {is_discrete}")

# Save cleaned results
clean_results = {
    'identity_results': identity_data,
    'summary': {
        'total_identity_experiments': len(identity_data),
        'dimensions_tested': [16, 64, 256],
        'regimes_tested': ['hard', 'near', 'smooth'],
        'performance_range': {
            'min_rate': min(r['rate'] for r in identity_data),
            'max_rate': max(r['rate'] for r in identity_data)
        }
    }
}

with open('results/diagnostics/processed_results.json', 'w') as f:
    json.dump(clean_results, f, indent=2)

print("\n✅ Results processed and saved to results/diagnostics/processed_results.json")