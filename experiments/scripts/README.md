# Figure and Table Generation Scripts

This directory contains publication-ready scripts for generating all figures and tables from experimental results.

## Scripts

### 1. `generate_figures.py`
Generates all publication-quality figures for the paper.

**Features:**
- Loads precomputed results from CSV/JSON/NPZ files
- Creates vector graphics (PDF/SVG) for LaTeX inclusion
- Supports multiple journal styles (default, Nature, IEEE)
- Generates both main and supplementary figures

**Key Figures Generated:**
- **Figure 1**: TVD convergence comparison (Klein vs IMHK)
- **Figure 2**: Dimension scaling analysis
- **Figure 3**: Parameter sensitivity plots
- **Figure 4**: Spectral gap scaling
- **Figure S1**: Basis reduction impact
- **Figure S2**: Center sensitivity
- **Figure S3**: Comprehensive lattice comparison
- **Figure S4**: Phase transition diagram

**Usage:**
```bash
# Generate all figures
python generate_figures.py

# Generate specific figures
python generate_figures.py --figures 1 2

# Use Nature journal style
python generate_figures.py --style nature

# Custom output directory
python generate_figures.py --output-dir custom/figures
```

### 2. `generate_tables.py`
Generates all publication-quality tables with proper formatting.

**Features:**
- Computes summary statistics with confidence intervals
- Outputs both LaTeX (.tex) and CSV formats
- Handles complex nested data structures
- Supports different journal formatting requirements

**Key Tables Generated:**
- **Table 1**: Cryptographic lattice benchmarks
- **Table 2**: Convergence properties summary
- **Table 3**: Dimension scaling analysis
- **Table 4**: Parameter sensitivity summary
- **Table S1**: Detailed basis reduction comparison
- **Table S2**: Spectral gap theoretical vs empirical
- **Table S3**: Computational timing breakdown
- **Table S4**: Theory vs empirical validation

**Usage:**
```bash
# Generate all tables
python generate_tables.py

# Generate specific tables
python generate_tables.py --tables 1 2

# Output only LaTeX format
python generate_tables.py --format tex

# Custom confidence level
python generate_tables.py --confidence 0.99
```

### 3. `run_all_experiments.py`
Master script that runs all experiments and generates results.

**Workflow:**
1. Runs convergence comparison experiments
2. Performs dimension scaling analysis
3. Executes cryptographic lattice benchmarks
4. Conducts parameter sensitivity analysis
5. Analyzes spectral gap behavior
6. Saves all results in machine-readable formats
7. Calls figure and table generation scripts

**Usage:**
```bash
# Run everything
python run_all_experiments.py

# Run with more CPU cores
python run_all_experiments.py --n-cores 8

# Run specific experiments only
python run_all_experiments.py --experiments convergence scaling
```

## Output Structure

When you run the scripts, they create the following structure:

```
paper/
├── figures/
│   ├── figure_1_convergence_comparison.pdf
│   ├── figure_1_convergence_comparison.svg
│   ├── figure_2_dimension_scaling.pdf
│   ├── figure_2_dimension_scaling.svg
│   ├── figure_3_parameter_sensitivity.pdf
│   ├── figure_3_parameter_sensitivity.svg
│   ├── figure_4_spectral_gap_scaling.pdf
│   ├── figure_4_spectral_gap_scaling.svg
│   ├── figure_manifest.json
│   └── figure_generation.log
└── tables/
    ├── table_1_cryptographic_benchmarks.tex
    ├── table_1_cryptographic_benchmarks.csv
    ├── table_2_convergence_summary.tex
    ├── table_2_convergence_summary.csv
    ├── table_3_dimension_scaling.tex
    ├── table_3_dimension_scaling.csv
    ├── table_4_parameter_sensitivity.tex
    ├── table_4_parameter_sensitivity.csv
    ├── table_manifest.json
    └── table_generation.log
```

## Data Requirements

The scripts expect experimental results in:
- `results/convergence_study/data/`
- `results/parameter_sensitivity/`

Results can be in formats:
- CSV files (preferred for simple data)
- JSON files (for nested structures)
- NPZ files (for large numpy arrays)

## Example LaTeX Usage

In your paper, include figures:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figures/figure_1_convergence_comparison}
    \caption{Convergence comparison between Klein and IMHK samplers.}
    \label{fig:convergence}
\end{figure}
```

Include tables:
```latex
\input{tables/table_1_cryptographic_benchmarks}
```

## Customization

### Adding New Figures

To add a new figure, edit `generate_figures.py`:

```python
def generate_figure_5_custom(self):
    """Generate custom figure."""
    # Load data
    data = self._load_custom_data()
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    
    # Save
    self._save_figure(fig, 'figure_5_custom', 
                     caption="My custom figure")
```

### Adding New Tables

To add a new table, edit `generate_tables.py`:

```python
def generate_table_5_custom(self):
    """Generate custom table."""
    # Load and process data
    data = self._load_custom_data()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save
    self._save_table(df, 'table_5_custom',
                    caption="My custom table",
                    label="tab:custom")
```

## Troubleshooting

1. **Missing data files**: Scripts will use synthetic data if real results aren't found
2. **LaTeX errors**: Ensure special characters are escaped in table cells
3. **Memory issues**: For large datasets, process in chunks or increase system memory
4. **Font issues**: Install required fonts or disable LaTeX rendering with `--style nature`

## Dependencies

Required packages:
- numpy, pandas, matplotlib, seaborn
- scipy (for statistics)
- Optional: LaTeX installation for text rendering

Install with:
```bash
pip install -r requirements.txt
```