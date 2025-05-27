#!/usr/bin/env python3
"""
Demo script to generate example outputs showing the figure and table generation capabilities.

This creates synthetic data and generates a few example figures and tables to demonstrate
the publication-quality output formatting.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from generate_figures import FigureGenerator
from generate_tables import TableGenerator
import os


def main():
    """Generate demo outputs."""
    print("Generating demo outputs...")
    
    # Create output directories
    demo_dir = Path("demo_outputs")
    demo_dir.mkdir(exist_ok=True)
    
    figures_dir = demo_dir / "figures"
    tables_dir = demo_dir / "tables"
    
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    
    # Generate demo figures
    print("\nGenerating demo figures...")
    fig_gen = FigureGenerator(
        results_dir="results",  # Will use synthetic data
        output_dir=str(figures_dir),
        style="default"
    )
    
    # Generate a few key figures
    try:
        print("  - Figure 1: Convergence comparison")
        fig_gen.generate_figure_1_convergence_comparison()
    except Exception as e:
        print(f"    (Using synthetic data: {e})")
    
    try:
        print("  - Figure 2: Dimension scaling")
        fig_gen.generate_figure_2_dimension_scaling()
    except Exception as e:
        print(f"    (Using synthetic data: {e})")
    
    # Generate demo tables
    print("\nGenerating demo tables...")
    table_gen = TableGenerator(
        results_dir="results",  # Will use synthetic data
        output_dir=str(tables_dir),
        style="default"
    )
    
    # Generate a few key tables
    try:
        print("  - Table 1: Cryptographic benchmarks")
        table_gen.generate_table_1_cryptographic_benchmarks()
    except Exception as e:
        print(f"    (Using synthetic data: {e})")
    
    try:
        print("  - Table 2: Convergence summary")
        table_gen.generate_table_2_convergence_summary()
    except Exception as e:
        print(f"    (Using synthetic data: {e})")
    
    # List generated files
    print("\nGenerated files:")
    
    print("\nFigures:")
    for f in sorted(figures_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print("\nTables:")
    for f in sorted(tables_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print(f"\nAll outputs saved to: {demo_dir.absolute()}")
    
    # Show a sample LaTeX table
    sample_tex = tables_dir / "table_1_cryptographic_benchmarks.tex"
    if sample_tex.exists():
        print("\nSample LaTeX table content:")
        print("-" * 60)
        with open(sample_tex, 'r') as f:
            print(f.read()[:500] + "...")
        print("-" * 60)


if __name__ == "__main__":
    main()