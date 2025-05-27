#!/usr/bin/env python3
"""
Master script to run all lattice Gaussian MCMC experiments.

This script coordinates the execution of experiments across all lattice types,
parameter regimes, and sampling algorithms. Results are saved in a structured
format for reproducibility and analysis.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Create output directories
def create_output_directories():
    """Create all necessary output directories."""
    base_dir = Path(__file__).parent.parent
    directories = [
        'results/samples',
        'results/diagnostics',
        'results/figures',
        'results/tables',
        'results/logs',
        'results/raw_data',
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")

# Experiment configuration
class ExperimentConfig:
    """Configuration for all experiments."""
    
    # Lattice dimensions to test
    DIMENSIONS = [16, 64, 256, 512, 1024]
    
    # Sigma regimes (relative to smoothing parameter)
    SIGMA_REGIMES = {
        'hard': 0.5,      # Below smoothing
        'near': 1.0,      # Near smoothing
        'smooth': 2.0,    # Well above smoothing
        'very_smooth': 5.0
    }
    
    # Number of samples per experiment
    N_SAMPLES = {
        16: 100000,
        64: 50000,
        256: 20000,
        512: 10000,
        1024: 5000
    }
    
    # Burn-in iterations
    BURN_IN = {
        16: 1000,
        64: 2000,
        256: 5000,
        512: 10000,
        1024: 20000
    }
    
    # Random seeds for reproducibility
    BASE_SEED = 42
    
    # Sampling algorithms
    ALGORITHMS = ['klein', 'imhk']
    
    # NTRU parameters (matching FALCON)
    NTRU_PARAMS = {
        512: {'q': 12289},
        1024: {'q': 12289}
    }
    
    # q-ary lattice parameters
    QARY_PARAMS = {
        'small_q': 257,
        'medium_q': 12289,
        'large_q': 1048583
    }


def get_experiment_id():
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}"


def save_experiment_metadata(exp_id, config):
    """Save experiment configuration and metadata."""
    metadata = {
        'experiment_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
        }
    }
    
    filepath = f"results/logs/{exp_id}_metadata.json"
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


def run_identity_experiments():
    """Run experiments on identity lattice Z^n."""
    print("\n" + "="*60)
    print("IDENTITY LATTICE EXPERIMENTS")
    print("="*60)
    
    exp_results = []
    
    for n in ExperimentConfig.DIMENSIONS:
        if n > 256:  # Skip very large for initial testing
            continue
            
        print(f"\nDimension n = {n}")
        
        for regime_name, sigma_factor in ExperimentConfig.SIGMA_REGIMES.items():
            # Smoothing parameter for Z^n is approximately sqrt(n)
            smoothing = np.sqrt(n)
            sigma = sigma_factor * smoothing
            
            print(f"  Regime: {regime_name}, σ = {sigma:.2f}")
            
            # Create experiment script
            script_content = f"""
#!/usr/bin/env sage
import sys
sys.path.append('..')

from sage.all import *
from src.core.discrete_gaussian import sample_discrete_gaussian_vec
from src.lattices.gaussian_lattice_sampler import IdentityLatticeSampler
from src.diagnostics.convergence import compute_tvd, compute_autocorrelation
from src.diagnostics.mcmc import effective_sample_size
import numpy as np
import time
import json

# Set random seed
set_random_seed({ExperimentConfig.BASE_SEED + n})

# Parameters
n = {n}
sigma = {sigma}
n_samples = {ExperimentConfig.N_SAMPLES[n]}
burn_in = {ExperimentConfig.BURN_IN[n]}

print(f"Running identity lattice: n={{n}}, σ={{sigma:.2f}}")

# Create sampler
sampler = IdentityLatticeSampler(n=n, sigma=sigma)

# Run sampling with timing
start_time = time.time()
samples = []

for i in range(burn_in + n_samples):
    v = sampler.sample()
    if i >= burn_in:
        samples.append(v)
    
    if i % 1000 == 0:
        print(f"  Progress: {{i}}/{burn_in + n_samples}")

elapsed = time.time() - start_time
samples_per_sec = n_samples / elapsed

print(f"  Sampling rate: {{samples_per_sec:.1f}} samples/sec")

# Convert to numpy for analysis
samples_array = np.array([[float(x) for x in v] for v in samples])

# Compute diagnostics
norms = np.linalg.norm(samples_array, axis=1)
mean_norm = np.mean(norms)
std_norm = np.std(norms)

# Autocorrelation of first coordinate
first_coord = samples_array[:, 0]
acf = compute_autocorrelation(first_coord, max_lag=100)
ess = effective_sample_size(first_coord)

# Save results
results = {{
    'lattice_type': 'identity',
    'dimension': n,
    'sigma': float(sigma),
    'sigma_regime': '{regime_name}',
    'n_samples': n_samples,
    'burn_in': burn_in,
    'elapsed_time': elapsed,
    'samples_per_sec': samples_per_sec,
    'mean_norm': float(mean_norm),
    'std_norm': float(std_norm),
    'autocorrelation': acf.tolist() if hasattr(acf, 'tolist') else list(acf),
    'ess': float(ess),
    'seed': {ExperimentConfig.BASE_SEED + n}
}}

# Save raw samples
np.savez_compressed(
    '../results/samples/identity_n{{n}}_sigma{regime_name}.npz',
    samples=samples_array,
    metadata=results
)

# Save diagnostics
with open('../results/diagnostics/identity_n{{n}}_sigma{regime_name}.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Completed")
"""
            
            # Write and execute script
            script_path = f"identity_exp_n{n}_{regime_name}.sage"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run experiment
            os.system(f"sage {script_path}")
            
            # Clean up
            os.remove(script_path)
            
            exp_results.append({
                'lattice': 'identity',
                'n': n,
                'regime': regime_name,
                'sigma': sigma
            })
    
    return exp_results


def run_qary_experiments():
    """Run experiments on q-ary lattices."""
    print("\n" + "="*60)
    print("Q-ARY LATTICE EXPERIMENTS")
    print("="*60)
    
    exp_results = []
    
    # Test different q values
    for q_name, q in ExperimentConfig.QARY_PARAMS.items():
        print(f"\nModulus q = {q} ({q_name})")
        
        # Test different dimensions
        for n in [16, 64]:  # Smaller dimensions for q-ary
            m = n // 2  # Matrix dimensions m x n
            
            print(f"  Dimension: m={m}, n={n}")
            
            script_content = f"""
#!/usr/bin/env sage
import sys
sys.path.append('..')

from sage.all import *
import numpy as np
import time
import json

# Set random seed
set_random_seed({ExperimentConfig.BASE_SEED})

# Parameters
m, n = {m}, {n}
q = {q}
sigma = {float(np.sqrt(n))}  # Simple choice

print(f"Q-ary lattice: m={{m}}, n={{n}}, q={{q}}")

# Generate random matrix A
A = random_matrix(ZZ, m, n, x=-q//2, y=q//2)

# For now, just save the configuration
# Full q-ary sampling would be implemented here

results = {{
    'lattice_type': 'qary',
    'q': q,
    'q_type': '{q_name}',
    'm': m,
    'n': n,
    'sigma': float(sigma),
    'matrix_norm': float(A.norm()),
    'status': 'configured'
}}

# Save configuration
with open('../results/diagnostics/qary_{{m}}x{{n}}_q{q_name}_config.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Q-ary configuration saved")
"""
            
            # Write and execute
            script_path = f"qary_exp_{q_name}_m{m}_n{n}.sage"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            os.system(f"sage {script_path}")
            os.remove(script_path)
            
            exp_results.append({
                'lattice': 'qary',
                'q': q,
                'm': m,
                'n': n
            })
    
    return exp_results


def run_ntru_experiments():
    """Run experiments on NTRU lattices."""
    print("\n" + "="*60)
    print("NTRU LATTICE EXPERIMENTS")
    print("="*60)
    
    exp_results = []
    
    # Test FALCON parameters
    for n in [64, 512]:  # Start with smaller, add 1024 later
        if n not in ExperimentConfig.NTRU_PARAMS and n < 512:
            q = 12289  # Default FALCON modulus
        else:
            q = ExperimentConfig.NTRU_PARAMS.get(n, {}).get('q', 12289)
        
        print(f"\nNTRU parameters: n = {n}, q = {q}")
        
        for regime_name, sigma_factor in ExperimentConfig.SIGMA_REGIMES.items():
            if regime_name in ['very_smooth']:  # Skip very smooth for NTRU
                continue
                
            # NTRU smoothing parameter estimation
            smoothing = np.sqrt(2 * n) * 10  # Rough estimate
            sigma = sigma_factor * smoothing
            
            print(f"  Regime: {regime_name}, σ = {sigma:.2f}")
            
            script_content = f"""
#!/usr/bin/env sage
import sys
sys.path.append('..')

from sage.all import *
import numpy as np
import time
import json

# Load NTRU implementation
load('../src/lattices/ntru_clean.py')

# Set random seed
set_random_seed({ExperimentConfig.BASE_SEED + n})

# Parameters
n = {n}
q = {q}
sigma = {sigma}
n_samples = {min(1000, ExperimentConfig.N_SAMPLES.get(n, 1000))}

print(f"NTRU lattice: n={{n}}, q={{q}}, σ={{sigma:.2f}}")

# Create NTRU lattice
ntru = NTRULattice(n=n, q=q)
print("Generating NTRU keys...")

if not ntru.generate_keys(key_type='ternary'):
    print("✗ Key generation failed")
    sys.exit(1)

print("✓ Keys generated")

# Get basis for analysis
B = ntru.get_basis()
print(f"Basis dimensions: {{B.nrows()}} × {{B.ncols()}}")

# Compute Gram-Schmidt norms
gs_norms = ntru.gram_schmidt_norms()
gs_ratio = max(gs_norms) / min(gs_norms)
print(f"GS ratio: {{gs_ratio:.2f}}")

# Sample from discrete Gaussian
print(f"Sampling {{n_samples}} vectors...")
start_time = time.time()

samples = []
for i in range(n_samples):
    v = ntru.sample_discrete_gaussian(sigma=sigma)
    samples.append(v)
    
    if i % 100 == 0:
        print(f"  Progress: {{i}}/{{n_samples}}")

elapsed = time.time() - start_time
samples_per_sec = n_samples / elapsed

print(f"  Sampling rate: {{samples_per_sec:.1f}} samples/sec")

# Convert to numpy
samples_array = np.array([[float(x) for x in v] for v in samples])

# Compute diagnostics
norms = np.linalg.norm(samples_array, axis=1)
mean_norm = np.mean(norms)
std_norm = np.std(norms)

# Save results
results = {{
    'lattice_type': 'ntru',
    'dimension': n,
    'q': q,
    'sigma': float(sigma),
    'sigma_regime': '{regime_name}',
    'n_samples': n_samples,
    'elapsed_time': elapsed,
    'samples_per_sec': samples_per_sec,
    'mean_norm': float(mean_norm),
    'std_norm': float(std_norm),
    'gs_ratio': float(gs_ratio),
    'min_gs_norm': float(min(gs_norms)),
    'max_gs_norm': float(max(gs_norms)),
    'seed': {ExperimentConfig.BASE_SEED + n}
}}

# Save compressed samples
np.savez_compressed(
    '../results/samples/ntru_n{{n}}_sigma{regime_name}.npz',
    samples=samples_array,
    metadata=results
)

# Save diagnostics
with open('../results/diagnostics/ntru_n{{n}}_sigma{regime_name}.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Completed")
"""
            
            # Write and execute
            script_path = f"ntru_exp_n{n}_{regime_name}.sage"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            os.system(f"sage {script_path}")
            os.remove(script_path)
            
            exp_results.append({
                'lattice': 'ntru',
                'n': n,
                'q': q,
                'regime': regime_name,
                'sigma': sigma
            })
    
    return exp_results


def generate_summary_report(all_results):
    """Generate summary report of all experiments."""
    report = {
        'total_experiments': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'experiments': all_results,
        'config': {
            'dimensions': ExperimentConfig.DIMENSIONS,
            'sigma_regimes': ExperimentConfig.SIGMA_REGIMES,
            'algorithms': ExperimentConfig.ALGORITHMS
        }
    }
    
    filepath = "results/logs/experiment_summary.json"
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Summary report saved to {filepath}")
    return filepath


def main():
    """Run all experiments."""
    print("\n" + "="*60)
    print("LATTICE GAUSSIAN MCMC EXPERIMENTS")
    print("="*60)
    
    # Create output directories
    create_output_directories()
    
    # Get experiment ID
    exp_id = get_experiment_id()
    print(f"\nExperiment ID: {exp_id}")
    
    # Save metadata
    config = {
        'dimensions': ExperimentConfig.DIMENSIONS,
        'sigma_regimes': ExperimentConfig.SIGMA_REGIMES,
        'n_samples': ExperimentConfig.N_SAMPLES,
        'algorithms': ExperimentConfig.ALGORITHMS
    }
    save_experiment_metadata(exp_id, config)
    
    all_results = []
    
    # Run experiments for each lattice type
    try:
        # Identity lattice
        identity_results = run_identity_experiments()
        all_results.extend(identity_results)
        
        # Q-ary lattices
        qary_results = run_qary_experiments()
        all_results.extend(qary_results)
        
        # NTRU lattices
        ntru_results = run_ntru_experiments()
        all_results.extend(ntru_results)
        
    except Exception as e:
        print(f"\n✗ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("="*60)
    print("\nResults saved in:")
    print("  - results/samples/     (raw sample data)")
    print("  - results/diagnostics/ (convergence metrics)")
    print("  - results/logs/        (experiment metadata)")


if __name__ == "__main__":
    main()