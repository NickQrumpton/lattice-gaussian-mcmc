
# Table Index for Lattice Gaussian MCMC Manuscript

## Main Paper Tables

### Table 1: Algorithm Performance Comparison
- **File**: `table_1_algorithm_comparison.tex`, `.csv`
- **Section**: Results - Algorithm Comparison
- **Description**: Performance comparison across sampling algorithms and lattice types
- **Key Metrics**: Sampling rate, ESS/n, mixing time, memory usage

### Table 2: Cryptographic Parameter Sets  
- **File**: `table_2_cryptographic_parameters.tex`, `.csv`
- **Section**: Background - Cryptographic Applications
- **Description**: Standard parameter sets for lattice-based cryptographic schemes
- **Key Metrics**: Security level, Hermite factor, key/signature sizes

### Table 3: Detailed Performance Benchmark
- **File**: `table_3_performance_benchmark.tex`, `.csv`
- **Section**: Results - Experimental Validation
- **Description**: Comprehensive performance data from experiments
- **Key Metrics**: Sample quality, error rates, autocorrelation

## Supplementary Tables

### Table S1: Convergence Analysis Summary
- **File**: `table_4_convergence_summary.tex`, `.csv`
- **Section**: Supplementary - Convergence Analysis
- **Description**: TVD, mixing times, spectral gaps for all algorithms
- **Key Metrics**: TVD convergence, integrated autocorrelation time

### Table S2: Computational Complexity Scaling
- **File**: `table_5_scaling_analysis.tex`, `.csv`
- **Section**: Supplementary - Complexity Analysis
- **Description**: Big-O scaling for different operations and lattice types
- **Key Metrics**: Time and space complexity bounds

## Usage Notes

1. **LaTeX Integration**: Include `.tex` files directly in manuscript with `\input{}`
2. **Data Analysis**: Use `.csv` files for further analysis or plotting
3. **Reproducibility**: All tables generated from experimental data in `results/diagnostics/`
4. **Updates**: Re-run `generate_tables.py` to refresh all tables from updated data

## Citation Format

Tables should be referenced as:
- Table 1: Main performance comparison
- Table 2: Cryptographic parameters (background)
- Table 3: Experimental validation
- Table S1: Convergence analysis (supplementary)
- Table S2: Scaling analysis (supplementary)
