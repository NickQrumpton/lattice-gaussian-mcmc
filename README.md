# Lattice Gaussian MCMC

Implementation of lattice Gaussian Markov Chain Monte Carlo (MCMC) methods based on Wang & Ling (2018) "On the Convergence of Lattice Gaussian Sampling".

## Overview

This repository provides a complete implementation of:
- **Klein's Algorithm**: Direct sampling from discrete Gaussian distributions on lattices
- **IMHK Algorithm**: Independent Metropolis-Hastings-Klein sampler with theoretical guarantees
- **Lattice Constructions**: Identity (Z^n), q-ary, NTRU, and custom lattices
- **Lattice Reduction**: LLL and BKZ algorithms optimized for sampling
- **Convergence Diagnostics**: Spectral gap analysis, TVD estimation, mixing time bounds
- **Visualization Tools**: Publication-quality plotting for lattice Gaussian research

## Installation

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Optional: SageMath for advanced lattice computations

### Install from source
```bash
git clone https://github.com/yourusername/lattice-gaussian-mcmc.git
cd lattice-gaussian-mcmc
pip install -e .
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Sampling
```python
from src.lattices.identity import IdentityLattice
from src.samplers.klein import KleinSampler

# Create a lattice (Z^10)
lattice = IdentityLattice(dimension=10)

# Create Klein sampler with � = 5
sampler = KleinSampler(lattice, sigma=5.0)

# Generate a sample
sample = sampler.sample()
print(f"Sample: {sample}")
```

### MCMC Sampling with IMHK
```python
from src.samplers.imhk import IMHKSampler

# Create IMHK sampler
imhk = IMHKSampler(lattice, sigma=5.0)

# Generate 10000 samples with burn-in
samples = imhk.sample(n_samples=10000, burn_in=1000, thin=10)
print(f"Generated {len(samples)} samples")
print(f"Acceptance rate: {imhk.acceptance_rate:.3f}")
```

### Cryptographic Lattices
```python
from src.lattices.qary import QaryLattice
from src.lattices.ntru import NTRULattice

# Create q-ary lattice for LWE
n, m, q = 256, 512, 3329
lwe_lattice = QaryLattice.from_lwe_instance(n, m, q, alpha=3.2/np.sqrt(q))

# Create NTRU lattice
ntru = NTRULattice(n=512, q=12289, sigma=4.05)
ntru.generate_basis()
```

### Lattice Reduction
```python
from src.lattices.reduction import LatticeReduction

# Reduce basis for better sampling
reducer = LatticeReduction()
reduced_basis, stats = reducer.sampling_reduce(
    lattice.get_basis(),
    target_distribution='gaussian',
    sigma=10.0
)

print(f"Sampling improvement: {stats['sampling_improvement']:.2f}x")
```

## Running Experiments

### Run all paper experiments
```bash
python experiments/scripts/run_all_experiments.py --output-dir results
```

### Run specific experiments
```bash
# Convergence comparison
python experiments/scripts/run_all_experiments.py --experiments convergence

# Dimension scaling analysis
python experiments/scripts/run_all_experiments.py --experiments scaling

# Cryptographic benchmarks
python experiments/scripts/run_all_experiments.py --experiments crypto
```

### Generate figures and tables
```bash
python experiments/scripts/generate_figures.py
python experiments/scripts/generate_tables.py
```

## Project Structure

```
lattice-gaussian-mcmc/
   src/
      core/           # Core abstractions
      lattices/       # Lattice implementations
      samplers/       # Sampling algorithms
      diagnostics/    # Convergence diagnostics
      visualization/  # Plotting utilities
   experiments/        # Experiment scripts
   tests/             # Test suites
   notebooks/         # Jupyter notebooks
   paper/             # LaTeX paper and figures
   results/           # Experimental results
```

## Key Components

### Lattices (`src/lattices/`)
- `base.py`: Abstract lattice interface
- `identity.py`: Integer lattice Z^n
- `qary.py`: q-ary lattices for cryptography
- `ntru.py`: NTRU lattice construction
- `reduction.py`: LLL/BKZ reduction algorithms

### Samplers (`src/samplers/`)
- `klein.py`: Klein's algorithm (Algorithm 1)
- `imhk.py`: Independent MH-Klein (Algorithm 2)
- `utils.py`: Discrete Gaussian utilities

### Diagnostics (`src/diagnostics/`)
- `spectral.py`: Spectral gap analysis
- `convergence.py`: TVD and mixing time estimation
- `mcmc.py`: General MCMC diagnostics

### Experiments (`experiments/`)
- `convergence_study.py`: Compare sampling algorithms
- `dimension_scaling.py`: Analyze scaling behavior
- `cryptographic_experiments.py`: Benchmark on crypto lattices
- `parameter_sensitivity.py`: Study parameter effects

## Theoretical Background

This implementation is based on:

**Wang, H., & Ling, S. (2018). On the convergence of lattice Gaussian sampling. IEEE Transactions on Information Theory.**

Key theoretical results implemented:
- Spectral gap bounds for IMHK (Theorem 1)
- Mixing time analysis (Theorem 2)
- Convergence guarantees for � e �_�(�)

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test modules:
```bash
pytest tests/test_lattices.py -v
pytest tests/test_samplers.py -v
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2018convergence,
  title={On the convergence of lattice {Gaussian} sampling},
  author={Wang, Huiwen and Ling, San},
  journal={IEEE Transactions on Information Theory},
  volume={64},
  number={11},
  pages={6646--6661},
  year={2018},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on theoretical work by Wang & Ling (2018)
- Inspired by implementations in the lattice cryptography community
- Uses algorithms from Micciancio & Regev's survey on lattice cryptography