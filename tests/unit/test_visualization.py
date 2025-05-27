"""
Unit tests for visualization and plotting utilities.

Tests cover plot generation, formatting, styling, and output validation
for all visualization components.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Dict, Any

from visualization.plots import PlottingTools


class TestPlottingTools:
    """Test the PlottingTools class and its methods."""
    
    def test_plotting_tools_initialization(self):
        """Test PlottingTools initialization."""
        plotter = PlottingTools()
        
        # Check that basic attributes exist
        assert hasattr(plotter, 'setup_publication_style')
        assert hasattr(plotter, 'plot_trace')
        assert hasattr(plotter, 'plot_autocorrelation')
        assert hasattr(plotter, 'plot_convergence_diagnostics')
        assert hasattr(plotter, 'plot_spectral_analysis')
        assert hasattr(plotter, 'plot_pairwise_scatter')
        assert hasattr(plotter, 'save_figure')
    
    def test_publication_style_setup(self):
        """Test publication style configuration."""
        plotter = PlottingTools()
        
        # Test different styles
        styles = ['ieee', 'nature', 'arxiv']
        
        for style in styles:
            plotter.setup_publication_style(style=style)
            
            # Check that matplotlib parameters are set
            current_params = plt.rcParams
            
            # Basic checks for publication quality
            assert current_params['font.size'] >= 10, "Font size should be readable"
            assert current_params['figure.dpi'] >= 100, "DPI should be high quality"
            assert 'serif' in current_params['font.family'] or 'sans-serif' in current_params['font.family']
    
    def test_invalid_style(self):
        """Test handling of invalid style parameters."""
        plotter = PlottingTools()
        
        with pytest.raises(ValueError, match="Unknown style"):
            plotter.setup_publication_style(style='invalid_style')
    
    def test_trace_plot_basic(self, temp_dir):
        """Test basic trace plot functionality."""
        plotter = PlottingTools()
        
        # Generate sample data
        n_samples = 1000
        chain_data = np.cumsum(np.random.randn(n_samples)) * 0.1
        
        # Create trace plot
        fig, ax = plotter.plot_trace(chain_data, title="Test Trace Plot")
        
        # Verify plot was created
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0, "Plot should contain data lines"
        
        # Check labels and title
        assert ax.get_title() == "Test Trace Plot"
        assert ax.get_xlabel() != "", "X-axis should be labeled"
        assert ax.get_ylabel() != "", "Y-axis should be labeled"
        
        # Save and verify file
        output_file = temp_dir / "test_trace.png"
        plotter.save_figure(fig, output_file, dpi=150)
        
        assert output_file.exists(), "Plot file should be saved"
        assert output_file.stat().st_size > 1000, "Plot file should not be empty"
        
        plt.close(fig)
    
    def test_trace_plot_multiple_chains(self, temp_dir):
        """Test trace plot with multiple chains."""
        plotter = PlottingTools()
        
        # Generate multiple chains
        n_samples = 500
        n_chains = 3
        
        chains = []
        for i in range(n_chains):
            chain = np.cumsum(np.random.randn(n_samples)) * 0.1 + i * 0.5
            chains.append(chain)
        
        # Create trace plot
        fig, ax = plotter.plot_trace(chains, title="Multiple Chains")
        
        # Verify all chains are plotted
        assert len(ax.lines) == n_chains, f"Should have {n_chains} lines"
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None, "Should have legend for multiple chains"
        
        plt.close(fig)
    
    def test_trace_plot_with_burn_in(self, temp_dir):
        """Test trace plot with burn-in highlighting."""
        plotter = PlottingTools()
        
        n_samples = 1000
        burn_in = 200
        chain_data = np.cumsum(np.random.randn(n_samples)) * 0.1
        
        fig, ax = plotter.plot_trace(chain_data, burn_in=burn_in, title="Trace with Burn-in")
        
        # Should have burn-in region highlighted
        # Check for axvline or shaded region indicating burn-in
        assert len(ax.collections) > 0 or len(ax.lines) > 1, \
            "Should have burn-in visualization"
        
        plt.close(fig)
    
    def test_autocorrelation_plot(self, temp_dir):
        """Test autocorrelation function plot."""
        plotter = PlottingTools()
        
        # Generate correlated data
        n_samples = 1000
        rho = 0.8
        data = np.zeros(n_samples)
        data[0] = np.random.randn()
        for i in range(1, n_samples):
            data[i] = rho * data[i-1] + np.sqrt(1-rho**2) * np.random.randn()
        
        # Create autocorrelation plot
        fig, ax = plotter.plot_autocorrelation(data, max_lag=50, title="Autocorrelation Test")
        
        # Verify plot structure
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0, "Should have autocorrelation line"
        
        # Check for confidence bounds
        lines = ax.lines
        assert len(lines) >= 2, "Should have confidence bounds"
        
        # Check axis labels
        assert "Lag" in ax.get_xlabel()
        assert "Autocorrelation" in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_autocorrelation_plot_edge_cases(self):
        """Test autocorrelation plot edge cases."""
        plotter = PlottingTools()
        
        # Very short series
        short_data = np.array([1, 2, 3])
        fig, ax = plotter.plot_autocorrelation(short_data, max_lag=10)
        
        # Should handle gracefully
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
        
        # Constant series
        constant_data = np.ones(100)
        fig, ax = plotter.plot_autocorrelation(constant_data, max_lag=20)
        
        # Should handle gracefully
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_convergence_diagnostics_plot(self, temp_dir):
        """Test convergence diagnostics visualization."""
        plotter = PlottingTools()
        
        # Generate sample diagnostic data
        n_samples = 1000
        iterations = np.arange(n_samples)
        
        diagnostics_data = {
            'gelman_rubin': 1.5 * np.exp(-iterations / 200) + 1.0,
            'effective_sample_size': iterations * 0.8,
            'geweke_scores': 0.1 * np.random.randn(n_samples)
        }
        
        # Create convergence plot
        fig, axes = plotter.plot_convergence_diagnostics(
            diagnostics_data, title="Convergence Diagnostics"
        )
        
        # Verify subplot structure
        assert fig is not None
        assert len(axes) >= 2, "Should have multiple diagnostic subplots"
        
        # Check that each subplot has data
        for ax in axes:
            assert len(ax.lines) > 0 or len(ax.collections) > 0, \
                "Each subplot should have data"
        
        plt.close(fig)
    
    def test_convergence_diagnostics_with_references(self):
        """Test convergence diagnostics with reference lines."""
        plotter = PlottingTools()
        
        diagnostics_data = {
            'gelman_rubin': np.array([1.5, 1.3, 1.15, 1.08, 1.05, 1.02]),
            'effective_sample_size': np.array([10, 25, 50, 80, 120, 150])
        }
        
        fig, axes = plotter.plot_convergence_diagnostics(
            diagnostics_data, 
            reference_lines={'gelman_rubin': 1.1, 'effective_sample_size': 100}
        )
        
        # Should have reference lines
        for ax in axes:
            # Check for horizontal reference lines
            h_lines = [line for line in ax.lines if len(set(line.get_ydata())) == 1]
            if len(h_lines) > 0:
                assert True  # Found reference line
                break
        else:
            assert False, "Should have reference lines"
        
        plt.close(fig)
    
    def test_spectral_analysis_plot(self, temp_dir):
        """Test spectral analysis visualization."""
        plotter = PlottingTools()
        
        # Generate sample spectral data
        n_eigenvalues = 50
        eigenvalues = np.sort(np.random.uniform(0, 1, n_eigenvalues))[::-1]
        eigenvalues[0] = 1.0  # Largest eigenvalue is 1
        
        spectral_data = {
            'eigenvalues': eigenvalues,
            'spectral_gap': eigenvalues[0] - eigenvalues[1],
            'mixing_time': 100.5
        }
        
        # Create spectral plot
        fig, axes = plotter.plot_spectral_analysis(
            spectral_data, title="Spectral Analysis"
        )
        
        # Verify plot structure
        assert fig is not None
        assert len(axes) >= 1, "Should have spectral plot"
        
        # Check eigenvalue plot
        main_ax = axes[0]
        assert len(main_ax.lines) > 0 or len(main_ax.collections) > 0, \
            "Should have eigenvalue data"
        
        # Check for spectral gap annotation
        annotations = main_ax.texts
        # Should have some text annotations for spectral gap
        
        plt.close(fig)
    
    def test_pairwise_scatter_plot(self, temp_dir):
        """Test pairwise scatter plot for multidimensional data."""
        plotter = PlottingTools()
        
        # Generate 3D sample data
        n_samples = 500
        dim = 3
        
        # Generate correlated data
        mean = np.zeros(dim)
        cov = np.eye(dim) + 0.3 * np.ones((dim, dim))
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Create pairwise plot
        fig, axes = plotter.plot_pairwise_scatter(
            samples, title="Pairwise Scatter Plot"
        )
        
        # Verify plot structure
        assert fig is not None
        
        # Should have dim x dim subplots
        if isinstance(axes, np.ndarray):
            assert axes.shape == (dim, dim), f"Should have {dim}x{dim} subplot grid"
            
            # Check diagonal plots (histograms)
            for i in range(dim):
                diag_ax = axes[i, i]
                assert len(diag_ax.patches) > 0, f"Diagonal plot {i} should have histogram"
            
            # Check off-diagonal plots (scatter)
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        scatter_ax = axes[i, j]
                        assert len(scatter_ax.collections) > 0, \
                            f"Scatter plot ({i},{j}) should have data"
        
        plt.close(fig)
    
    def test_pairwise_scatter_2d(self):
        """Test pairwise scatter for 2D data."""
        plotter = PlottingTools()
        
        # Generate 2D data
        n_samples = 300
        samples = np.random.randn(n_samples, 2)
        
        fig, axes = plotter.plot_pairwise_scatter(samples)
        
        # For 2D, should create a 2x2 grid
        assert fig is not None
        
        plt.close(fig)
    
    def test_save_figure_different_formats(self, temp_dir):
        """Test saving figures in different formats."""
        plotter = PlottingTools()
        
        # Create simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        
        # Test different formats
        formats = ['png', 'pdf', 'svg', 'eps']
        
        for fmt in formats:
            output_file = temp_dir / f"test_plot.{fmt}"
            plotter.save_figure(fig, output_file, dpi=150)
            
            assert output_file.exists(), f"File should be saved in {fmt} format"
            assert output_file.stat().st_size > 0, f"{fmt} file should not be empty"
        
        plt.close(fig)
    
    def test_save_figure_with_options(self, temp_dir):
        """Test saving figures with various options."""
        plotter = PlottingTools()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.random.randn(100))
        
        output_file = temp_dir / "test_with_options.png"
        
        # Save with specific options
        plotter.save_figure(
            fig, output_file, 
            dpi=300, 
            bbox_inches='tight',
            transparent=True,
            facecolor='white'
        )
        
        assert output_file.exists()
        
        # Check file size is reasonable for high DPI
        assert output_file.stat().st_size > 10000, "High DPI file should be larger"
        
        plt.close(fig)
    
    @pytest.mark.edge_case
    def test_plotting_edge_cases(self):
        """Test plotting functions with edge case inputs."""
        plotter = PlottingTools()
        
        # Empty data
        fig, ax = plotter.plot_trace(np.array([]))
        assert fig is not None
        assert ax is not None
        plt.close(fig)
        
        # Single point
        fig, ax = plotter.plot_trace(np.array([1.0]))
        assert fig is not None
        plt.close(fig)
        
        # NaN values
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        fig, ax = plotter.plot_trace(data_with_nan)
        assert fig is not None
        plt.close(fig)
        
        # Infinite values
        data_with_inf = np.array([1, 2, np.inf, 4, 5])
        fig, ax = plotter.plot_trace(data_with_inf)
        assert fig is not None
        plt.close(fig)
    
    def test_figure_size_and_layout(self):
        """Test figure sizing and layout options."""
        plotter = PlottingTools()
        
        # Test custom figure size
        data = np.random.randn(100)
        
        fig, ax = plotter.plot_trace(data, figsize=(12, 8))
        
        # Check figure size
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        
        plt.close(fig)
        
        # Test tight layout
        fig, axes = plotter.plot_convergence_diagnostics({
            'gelman_rubin': np.random.rand(50) + 1,
            'effective_sample_size': np.random.rand(50) * 100
        }, tight_layout=True)
        
        # Should not raise layout warnings
        assert fig is not None
        
        plt.close(fig)
    
    def test_color_schemes_and_styling(self):
        """Test different color schemes and styling options."""
        plotter = PlottingTools()
        
        # Test with different color schemes
        data = [np.random.randn(100) + i for i in range(3)]
        
        # Default colors
        fig, ax = plotter.plot_trace(data, title="Default Colors")
        colors_default = [line.get_color() for line in ax.lines]
        plt.close(fig)
        
        # Custom colors
        custom_colors = ['red', 'blue', 'green']
        fig, ax = plotter.plot_trace(data, colors=custom_colors, title="Custom Colors")
        colors_custom = [line.get_color() for line in ax.lines]
        
        # Colors should be different from default
        assert colors_custom != colors_default
        
        plt.close(fig)
    
    def test_latex_rendering(self):
        """Test LaTeX rendering in plots if available."""
        plotter = PlottingTools()
        
        # Test with LaTeX-style labels
        data = np.random.randn(100)
        
        fig, ax = plotter.plot_trace(
            data, 
            title=r"Test with $\sigma = 1.0$",
            xlabel=r"Iteration $t$",
            ylabel=r"$X_t$"
        )
        
        # Should not raise errors
        assert fig is not None
        assert ax is not None
        
        # Check that labels are set
        assert ax.get_title() != ""
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        
        plt.close(fig)


class TestPlotValidation:
    """Test plot content validation and quality checks."""
    
    def test_plot_data_integrity(self):
        """Test that plotted data matches input data."""
        plotter = PlottingTools()
        
        # Generate test data
        input_data = np.array([1, 4, 2, 8, 5, 7])
        
        fig, ax = plotter.plot_trace(input_data)
        
        # Extract plotted data
        line = ax.lines[0]
        plotted_y = line.get_ydata()
        
        # Should match input data
        np.testing.assert_array_equal(plotted_y, input_data)
        
        plt.close(fig)
    
    def test_axis_limits_and_scaling(self):
        """Test appropriate axis limits and scaling."""
        plotter = PlottingTools()
        
        # Test with data of different scales
        small_data = np.random.randn(100) * 1e-6
        large_data = np.random.randn(100) * 1e6
        
        # Small data
        fig, ax = plotter.plot_trace(small_data)
        y_lim = ax.get_ylim()
        data_range = np.max(small_data) - np.min(small_data)
        
        # Axis limits should be reasonable for data range
        axis_range = y_lim[1] - y_lim[0]
        assert axis_range > data_range, "Axis range should encompass data"
        assert axis_range < data_range * 100, "Axis range should not be too large"
        
        plt.close(fig)
        
        # Large data
        fig, ax = plotter.plot_trace(large_data)
        y_lim = ax.get_ylim()
        data_range = np.max(large_data) - np.min(large_data)
        axis_range = y_lim[1] - y_lim[0]
        
        assert axis_range > data_range, "Axis range should encompass data"
        
        plt.close(fig)
    
    def test_grid_and_styling_consistency(self):
        """Test grid lines and styling consistency."""
        plotter = PlottingTools()
        
        # Setup publication style
        plotter.setup_publication_style(style='ieee')
        
        data = np.random.randn(100)
        fig, ax = plotter.plot_trace(data, grid=True)
        
        # Check grid is enabled
        assert ax.grid(None), "Grid should be enabled when requested"
        
        plt.close(fig)
    
    def test_legend_and_labels_presence(self):
        """Test presence and quality of legends and labels."""
        plotter = PlottingTools()
        
        # Multiple series with labels
        data1 = np.random.randn(100)
        data2 = np.random.randn(100) + 1
        
        fig, ax = plotter.plot_trace(
            [data1, data2], 
            labels=['Series 1', 'Series 2'],
            title="Test Plot",
            xlabel="X Label",
            ylabel="Y Label"
        )
        
        # Check title and axis labels
        assert ax.get_title() == "Test Plot"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None, "Should have legend for multiple series"
        
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert 'Series 1' in legend_texts
        assert 'Series 2' in legend_texts
        
        plt.close(fig)


@pytest.mark.integration
class TestVisualizationIntegration:
    """Test visualization integration with other components."""
    
    def test_plotting_with_real_mcmc_data(self, temp_dir):
        """Test plotting with real MCMC chain data."""
        from lattices.identity import IdentityLattice
        from samplers.imhk import IMHKSampler
        
        plotter = PlottingTools()
        
        # Generate real MCMC data
        lattice = IdentityLattice(dimension=2)
        sampler = IMHKSampler(lattice, sigma=1.0)
        
        # Burn-in
        for _ in range(200):
            sampler.sample()
        
        # Collect samples
        n_samples = 1000
        samples = np.array([sampler.sample() for _ in range(n_samples)])
        
        # Test trace plot
        fig, ax = plotter.plot_trace(samples[:, 0], title="MCMC Trace")
        output_file = temp_dir / "mcmc_trace.png"
        plotter.save_figure(fig, output_file)
        assert output_file.exists()
        plt.close(fig)
        
        # Test pairwise scatter
        fig, axes = plotter.plot_pairwise_scatter(samples, title="MCMC Samples")
        output_file = temp_dir / "mcmc_pairwise.png"
        plotter.save_figure(fig, output_file)
        assert output_file.exists()
        plt.close(fig)
        
        # Test autocorrelation
        fig, ax = plotter.plot_autocorrelation(samples[:, 0], title="MCMC Autocorrelation")
        output_file = temp_dir / "mcmc_autocorr.png"
        plotter.save_figure(fig, output_file)
        assert output_file.exists()
        plt.close(fig)
    
    def test_plotting_with_diagnostic_data(self, temp_dir):
        """Test plotting with real diagnostic data."""
        from diagnostics.convergence import ConvergenceDiagnostics
        
        plotter = PlottingTools()
        diagnostics = ConvergenceDiagnostics()
        
        # Generate sample chains
        n_samples = 500
        n_chains = 3
        chains = [np.cumsum(np.random.randn(n_samples)) * 0.1 for _ in range(n_chains)]
        
        # Compute real diagnostics
        rhat = diagnostics.gelman_rubin_diagnostic(chains)
        ess_values = [diagnostics.effective_sample_size(chain) for chain in chains]
        
        # Create diagnostic data
        diagnostic_data = {
            'gelman_rubin': np.array([rhat] * 10),  # Simplified for plotting
            'effective_sample_size': np.array(ess_values[:10])  # Take first 10
        }
        
        # Plot diagnostics
        fig, axes = plotter.plot_convergence_diagnostics(
            diagnostic_data, 
            title="Real Diagnostic Data"
        )
        
        output_file = temp_dir / "real_diagnostics.png"
        plotter.save_figure(fig, output_file)
        assert output_file.exists()
        plt.close(fig)


@pytest.mark.performance
class TestVisualizationPerformance:
    """Test visualization performance and memory usage."""
    
    def test_large_dataset_plotting(self, performance_config):
        """Test plotting performance with large datasets."""
        import time
        
        plotter = PlottingTools()
        
        # Large dataset
        n_samples = 50000
        large_data = np.cumsum(np.random.randn(n_samples)) * 0.1
        
        # Time the plotting
        start_time = time.time()
        fig, ax = plotter.plot_trace(large_data, title="Large Dataset")
        plot_time = time.time() - start_time
        
        # Should complete in reasonable time
        max_time = performance_config['max_time_simple'] * 2  # Allow more time for large plots
        assert plot_time < max_time, f"Plotting too slow for large dataset: {plot_time:.2f}s"
        
        plt.close(fig)
    
    def test_memory_usage_plotting(self):
        """Test memory usage doesn't explode with plotting."""
        import gc
        
        plotter = PlottingTools()
        
        # Create many plots to test memory cleanup
        for i in range(10):
            data = np.random.randn(1000)
            fig, ax = plotter.plot_trace(data, title=f"Plot {i}")
            plt.close(fig)
            
            # Force garbage collection
            gc.collect()
        
        # If we get here without memory errors, test passes
        assert True


@pytest.mark.reproducibility
class TestVisualizationReproducibility:
    """Test reproducibility of visualization outputs."""
    
    def test_plot_determinism(self, temp_dir, test_seed):
        """Test that plots are deterministic given same input."""
        plotter = PlottingTools()
        
        # Generate deterministic data
        np.random.seed(test_seed)
        data = np.random.randn(100)
        
        # Create two identical plots
        fig1, ax1 = plotter.plot_trace(data, title="Deterministic Plot")
        fig2, ax2 = plotter.plot_trace(data, title="Deterministic Plot")
        
        # Extract plot data
        line1_data = ax1.lines[0].get_ydata()
        line2_data = ax2.lines[0].get_ydata()
        
        # Should be identical
        np.testing.assert_array_equal(line1_data, line2_data)
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_save_reproducibility(self, temp_dir, test_seed):
        """Test that saved figures are reproducible."""
        plotter = PlottingTools()
        
        np.random.seed(test_seed)
        data = np.random.randn(100)
        
        # Save same plot twice
        for i in range(2):
            fig, ax = plotter.plot_trace(data, title="Reproducible Save")
            output_file = temp_dir / f"repro_test_{i}.png"
            plotter.save_figure(fig, output_file, dpi=150)
            plt.close(fig)
        
        # Files should exist and have same size (approximately)
        file1 = temp_dir / "repro_test_0.png"
        file2 = temp_dir / "repro_test_1.png"
        
        assert file1.exists() and file2.exists()
        
        size1 = file1.stat().st_size
        size2 = file2.stat().st_size
        
        # Sizes should be very close (exact match depends on compression)
        size_diff_ratio = abs(size1 - size2) / max(size1, size2)
        assert size_diff_ratio < 0.01, "Saved file sizes should be nearly identical"