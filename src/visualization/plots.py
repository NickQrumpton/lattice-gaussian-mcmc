"""
PlottingTools: Robust plotting module for publication-quality figures.

Produces consistent, clear figures for lattice Gaussian MCMC research,
matching or exceeding the visual standards of Wang & Ling (2018).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib import cm, colors
from matplotlib.ticker import LogLocator, MultipleLocator
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
import os
import json
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd


class PlottingTools:
    """
    Comprehensive plotting tools for lattice Gaussian MCMC visualization.
    
    Ensures consistent style, publication quality, and reproducibility.
    """
    
    # Standard figure sizes (inches)
    SINGLE_COL_WIDTH = 3.5
    DOUBLE_COL_WIDTH = 7.0
    GOLDEN_RATIO = 1.618
    
    # Color schemes
    COLORBLIND_PALETTE = [
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
        '#a65628', '#984ea3', '#999999', '#e41a1c', 
        '#dede00', '#377eb8'
    ]
    
    def __init__(self, style='publication', use_latex=True, dpi=300):
        """
        Initialize plotting tools with consistent style.
        
        Args:
            style: Plotting style ('publication', 'presentation', 'draft')
            use_latex: Enable LaTeX rendering
            dpi: Resolution for raster outputs
        """
        self.style = style
        self.use_latex = use_latex
        self.dpi = dpi
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
        
        # Create output directories
        self.figure_dir = '../results/figures'
        os.makedirs(self.figure_dir, exist_ok=True)
        
    def _setup_matplotlib_style(self):
        """Configure matplotlib for publication quality."""
        # Base settings
        plt.style.use('seaborn-v0_8-paper')
        
        # Font settings
        if self.style == 'publication':
            font_size = 11
            tick_size = 10
            legend_size = 10
        elif self.style == 'presentation':
            font_size = 14
            tick_size = 12
            legend_size = 12
        else:  # draft
            font_size = 12
            tick_size = 11
            legend_size = 11
            
        # Apply settings
        plt.rcParams.update({
            # Font
            'font.size': font_size,
            'axes.titlesize': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'legend.fontsize': legend_size,
            
            # LaTeX
            'text.usetex': self.use_latex,
            'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm}',
            
            # Figure
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Lines
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'patch.linewidth': 0.5,
            
            # Axes
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            
            # Colors
            'axes.prop_cycle': plt.cycler('color', self.COLORBLIND_PALETTE),
            
            # Legend
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.borderpad': 0.5,
            
            # Ticks
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
        })
        
    # ========== 1. Consistent Style Helpers ==========
    
    def get_figure_size(self, width='single', aspect_ratio=None):
        """
        Get consistent figure size.
        
        Args:
            width: 'single' or 'double' column
            aspect_ratio: Height/width ratio (default: 1/golden_ratio)
            
        Returns:
            tuple: (width, height) in inches
        """
        if width == 'single':
            w = self.SINGLE_COL_WIDTH
        elif width == 'double':
            w = self.DOUBLE_COL_WIDTH
        else:
            w = float(width)
            
        if aspect_ratio is None:
            aspect_ratio = 1.0 / self.GOLDEN_RATIO
            
        h = w * aspect_ratio
        return (w, h)
    
    def set_axis_style(self, ax, xlabel=None, ylabel=None, title=None,
                      xlim=None, ylim=None, legend=True):
        """Apply consistent axis styling."""
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Legend
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', frameon=True, framealpha=0.8,
                     edgecolor='black', borderpad=0.5)
            
    # ========== 2. Paper-Style Figure Recreation ==========
    
    def plot_lattice_gaussian_2d(self, lattice_points, sigma, center=None,
                                n_contours=20, show_samples=True,
                                save_name=None, width='single'):
        """
        Create 2D heatmap/contour plot of lattice Gaussian (Figure 1 style).
        
        Args:
            lattice_points: 2D array of lattice points
            sigma: Standard deviation
            center: Center of Gaussian (default: origin)
            n_contours: Number of contour levels
            show_samples: Overlay sample points
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        if center is None:
            center = np.zeros(2)
            
        # Create grid for heatmap
        x_range = lattice_points[:, 0]
        y_range = lattice_points[:, 1]
        x_min, x_max = x_range.min() - 3*sigma, x_range.max() + 3*sigma
        y_min, y_max = y_range.min() - 3*sigma, y_range.max() + 3*sigma
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        
        # Compute Gaussian density
        pos = np.dstack((xx, yy))
        rv = stats.multivariate_normal(center, sigma**2 * np.eye(2))
        density = rv.pdf(pos)
        
        # Plot heatmap
        im = ax.contourf(xx, yy, density, levels=n_contours, cmap='viridis')
        
        # Add contour lines
        contours = ax.contour(xx, yy, density, levels=n_contours,
                             colors='white', alpha=0.4, linewidths=0.5)
        
        # Plot lattice points
        if show_samples:
            # Weight by Gaussian
            weights = rv.pdf(lattice_points)
            weights = weights / weights.max()
            
            # Plot with size proportional to weight
            scatter = ax.scatter(lattice_points[:, 0], lattice_points[:, 1],
                               s=50 * weights, c='red', alpha=0.7,
                               edgecolors='darkred', linewidth=0.5)
                               
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\rho_{\sigma,c}(x)$')
        
        # Styling
        self.set_axis_style(ax, xlabel='$x_1$', ylabel='$x_2$',
                           title='Discrete Gaussian on Lattice')
        ax.set_aspect('equal')
        
        # Save
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    def plot_convergence_comparison(self, results_dict, save_name=None,
                                   width='double'):
        """
        Plot convergence comparison (Figure 2 style).
        
        Args:
            results_dict: Dictionary with algorithm results
            save_name: Filename to save
            width: Figure width
        """
        fig, axes = plt.subplots(2, 2, figsize=self.get_figure_size(width, 0.8))
        
        # Extract data
        algorithms = list(results_dict.keys())
        colors = self.COLORBLIND_PALETTE[:len(algorithms)]
        
        # Plot 1: TVD vs iterations
        ax = axes[0, 0]
        for algo, color in zip(algorithms, colors):
            data = results_dict[algo]
            if 'tvd_history' in data:
                iterations = data['tvd_history']['iterations']
                tvd = data['tvd_history']['tvd']
                tvd_lower = data['tvd_history'].get('ci_lower', tvd)
                tvd_upper = data['tvd_history'].get('ci_upper', tvd)
                
                ax.semilogy(iterations, tvd, color=color, label=algo, linewidth=1.5)
                ax.fill_between(iterations, tvd_lower, tvd_upper,
                              color=color, alpha=0.2)
                              
        # Theoretical bound
        if 'theoretical_tvd' in results_dict[algorithms[0]]:
            t_max = max(data['tvd_history']['iterations'])
            t_theory = np.linspace(1, t_max, 100)
            delta = results_dict[algorithms[0]]['delta']
            tvd_theory = (1 - delta) ** t_theory
            ax.semilogy(t_theory, tvd_theory, 'k--', label='Theoretical',
                       linewidth=1.0, alpha=0.7)
                       
        self.set_axis_style(ax, xlabel='Iteration $t$', 
                           ylabel=r'$\|P^t(x,\cdot) - \pi\|_{TV}$',
                           title='Total Variation Distance')
                           
        # Plot 2: Acceptance rate
        ax = axes[0, 1]
        for algo, color in zip(algorithms, colors):
            data = results_dict[algo]
            if 'acceptance_history' in data:
                iterations = data['acceptance_history']['iterations']
                rates = data['acceptance_history']['rates']
                ax.plot(iterations, rates, color=color, label=algo, linewidth=1.5)
                
        self.set_axis_style(ax, xlabel='Iteration $t$',
                           ylabel='Acceptance Rate',
                           title='MH Acceptance Probability')
        ax.set_ylim([0, 1])
        
        # Plot 3: Mixing indicator
        ax = axes[1, 0]
        for algo, color in zip(algorithms, colors):
            data = results_dict[algo]
            if 'autocorrelation' in data:
                lags = data['autocorrelation']['lags']
                acf = data['autocorrelation']['acf']
                ax.plot(lags, acf, color=color, label=algo, linewidth=1.5)
                
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=0.5,
                  label=r'$\rho = 0.05$')
        self.set_axis_style(ax, xlabel='Lag', ylabel='ACF',
                           title='Autocorrelation Function')
                           
        # Plot 4: Spectral gap comparison
        ax = axes[1, 1]
        spectral_data = []
        for algo in algorithms:
            if 'spectral_gap' in results_dict[algo]:
                spectral_data.append({
                    'Algorithm': algo,
                    'Theoretical': results_dict[algo]['spectral_gap']['theoretical'],
                    'Empirical': results_dict[algo]['spectral_gap']['empirical']
                })
                
        if spectral_data:
            df = pd.DataFrame(spectral_data)
            x = np.arange(len(spectral_data))
            width_bar = 0.35
            
            ax.bar(x - width_bar/2, df['Theoretical'], width_bar,
                  label='Theoretical', color=colors[0], alpha=0.8)
            ax.bar(x + width_bar/2, df['Empirical'], width_bar,
                  label='Empirical', color=colors[1], alpha=0.8)
                  
            ax.set_xticks(x)
            ax.set_xticklabels(df['Algorithm'])
            self.set_axis_style(ax, ylabel='Spectral Gap $\gamma$',
                               title='Spectral Gap Comparison')
                               
        plt.tight_layout()
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, axes
    
    def plot_delta_scaling(self, dimensions, delta_inverse_values,
                          sigma_info=None, save_name=None, width='single'):
        """
        Plot 1/´ scaling with dimension (Figure 3 style).
        
        Args:
            dimensions: Array of dimensions
            delta_inverse_values: Dictionary of 1/´ values by method
            sigma_info: Information about Ã parameters
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        # Plot each method
        for i, (method, values) in enumerate(delta_inverse_values.items()):
            color = self.COLORBLIND_PALETTE[i]
            ax.semilogy(dimensions, values, 'o-', color=color,
                       label=method, linewidth=1.5, markersize=6)
                       
        # Add growth rate annotation if available
        if len(dimensions) > 3:
            # Fit exponential growth
            log_vals = np.log(list(delta_inverse_values.values())[0])
            slope = np.polyfit(dimensions, log_vals, 1)[0]
            growth_rate = np.exp(slope)
            
            ax.text(0.05, 0.95, f'Growth rate: {growth_rate:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                   
        # Add Ã information
        if sigma_info:
            info_text = f"$\sigma^2 = {sigma_info['ratio']:.2f} \cdot \min_i \|b^*_i\|^2$"
            ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                   
        self.set_axis_style(ax, xlabel='Dimension $n$',
                           ylabel=r'$1/\delta$',
                           title='Inverse Spectral Gap Scaling')
                           
        # Minor ticks for log scale
        ax.yaxis.set_minor_locator(LogLocator(subs='all'))
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    # ========== 3. Core Plotting Functions ==========
    
    def plot_chain_trace(self, chain, dimensions=(0, 1), burn_in=0,
                        color_by='iteration', save_name=None, width='single'):
        """
        Visualize MCMC chain trajectory in 2D projection.
        
        Args:
            chain: MCMC samples (n_samples x n_dims)
            dimensions: Which two dimensions to plot
            burn_in: Number of burn-in samples to highlight
            color_by: 'iteration' or 'density'
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        # Extract dimensions
        x = chain[:, dimensions[0]]
        y = chain[:, dimensions[1]]
        
        if color_by == 'iteration':
            # Color by iteration number
            colors = np.arange(len(chain))
            scatter = ax.scatter(x[burn_in:], y[burn_in:], c=colors[burn_in:],
                               cmap='viridis', s=20, alpha=0.6)
                               
            # Highlight burn-in
            if burn_in > 0:
                ax.scatter(x[:burn_in], y[:burn_in], c='red', s=10,
                          alpha=0.3, label='Burn-in')
                          
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Iteration')
            
        else:  # density
            # Compute 2D histogram
            hist, xedges, yedges = np.histogram2d(x, y, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax.imshow(hist.T, origin='lower', extent=extent,
                          cmap='Blues', aspect='auto')
                          
            # Overlay trajectory
            ax.plot(x, y, 'r-', alpha=0.3, linewidth=0.5)
            ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
            ax.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Sample Density')
            
        self.set_axis_style(ax, xlabel=f'Dimension {dimensions[0]}',
                           ylabel=f'Dimension {dimensions[1]}',
                           title='MCMC Chain Trajectory')
                           
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    def plot_autocorrelation(self, chain, max_lag=None, confidence=True,
                           save_name=None, width='single'):
        """
        Plot autocorrelation function with confidence intervals.
        
        Args:
            chain: Time series data
            max_lag: Maximum lag to plot
            confidence: Show confidence intervals
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        # Compute ACF
        if max_lag is None:
            max_lag = min(len(chain) // 4, 100)
            
        acf_values = []
        ci_lower = []
        ci_upper = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_values.append(1.0)
                ci_lower.append(1.0)
                ci_upper.append(1.0)
            else:
                acf = self._compute_acf(chain, lag)
                acf_values.append(acf)
                
                # Confidence intervals (approximate)
                if confidence:
                    se = 1.0 / np.sqrt(len(chain))
                    ci_lower.append(acf - 1.96 * se)
                    ci_upper.append(acf + 1.96 * se)
                    
        lags = np.arange(max_lag + 1)
        
        # Plot ACF
        ax.stem(lags[1:], acf_values[1:], linefmt='b-', markerfmt='bo',
               basefmt='k-', label='ACF')
               
        # Confidence intervals
        if confidence:
            ax.fill_between(lags[1:], ci_lower[1:], ci_upper[1:],
                           color='blue', alpha=0.2, label='95% CI')
                           
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=0.5,
                  label=r'$\rho = 0.05$')
        ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=0.5)
        
        self.set_axis_style(ax, xlabel='Lag', ylabel='ACF',
                           title='Autocorrelation Function',
                           ylim=[-0.2, 1.1])
                           
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    # ========== 4. Lattice Visualization ==========
    
    def plot_lattice_points(self, basis, n_points=5, gaussian_overlay=True,
                          sigma=1.0, dim_3d=False, save_name=None, width='single'):
        """
        Visualize lattice points with optional Gaussian overlay.
        
        Args:
            basis: Lattice basis matrix
            n_points: Number of points in each direction
            gaussian_overlay: Add Gaussian heatmap
            sigma: Gaussian standard deviation
            dim_3d: Use 3D plot for 3D lattices
            save_name: Filename to save
            width: Figure width
        """
        n_dim = basis.shape[0]
        
        if n_dim == 2:
            fig, ax = plt.subplots(figsize=self.get_figure_size(width))
            
            # Generate lattice points
            points = []
            for i in range(-n_points, n_points + 1):
                for j in range(-n_points, n_points + 1):
                    point = i * basis[0] + j * basis[1]
                    points.append(point)
                    
            points = np.array(points)
            
            # Plot lattice points
            ax.scatter(points[:, 0], points[:, 1], s=30, c='blue',
                      edgecolors='darkblue', linewidth=0.5, zorder=3)
                      
            # Add basis vectors
            origin = np.zeros(2)
            ax.arrow(origin[0], origin[1], basis[0, 0], basis[0, 1],
                    head_width=0.1, head_length=0.1, fc='red', ec='red',
                    linewidth=2, zorder=4, label='$b_1$')
            ax.arrow(origin[0], origin[1], basis[1, 0], basis[1, 1],
                    head_width=0.1, head_length=0.1, fc='green', ec='green',
                    linewidth=2, zorder=4, label='$b_2$')
                    
            # Gaussian overlay
            if gaussian_overlay:
                x_range = points[:, 0]
                y_range = points[:, 1]
                x_min, x_max = x_range.min() - sigma, x_range.max() + sigma
                y_min, y_max = y_range.min() - sigma, y_range.max() + sigma
                
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                    np.linspace(y_min, y_max, 200))
                                    
                rv = stats.multivariate_normal([0, 0], sigma**2 * np.eye(2))
                density = rv.pdf(np.dstack((xx, yy)))
                
                contours = ax.contour(xx, yy, density, levels=10,
                                     colors='gray', alpha=0.5, linewidths=0.5)
                                     
            self.set_axis_style(ax, xlabel='$x_1$', ylabel='$x_2$',
                               title='Lattice Structure')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, zorder=1)
            
        elif n_dim == 3 and dim_3d:
            fig = plt.figure(figsize=self.get_figure_size(width))
            ax = fig.add_subplot(111, projection='3d')
            
            # Generate lattice points
            points = []
            for i in range(-n_points, n_points + 1):
                for j in range(-n_points, n_points + 1):
                    for k in range(-n_points, n_points + 1):
                        point = i * basis[0] + j * basis[1] + k * basis[2]
                        points.append(point)
                        
            points = np.array(points)
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=20, c='blue', alpha=0.6)
                      
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$x_3$')
            ax.set_title('3D Lattice Structure')
            
        else:
            # For higher dimensions, project to 2D
            return self.plot_lattice_points(basis[:2, :2], n_points,
                                          gaussian_overlay, sigma, False,
                                          save_name, width)
                                          
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    def plot_voronoi_cells(self, basis, save_name=None, width='single'):
        """
        Plot Voronoi cells/fundamental domain for 2D lattice.
        
        Args:
            basis: 2x2 lattice basis
            save_name: Filename to save
            width: Figure width
        """
        if basis.shape[0] != 2:
            raise ValueError("Voronoi plot only supports 2D lattices")
            
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        # Generate lattice points
        points = []
        n_points = 5
        for i in range(-n_points, n_points + 1):
            for j in range(-n_points, n_points + 1):
                point = i * basis[0] + j * basis[1]
                points.append(point)
                
        points = np.array(points)
        
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Plot
        voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                       line_colors='blue', line_width=1)
                       
        # Highlight fundamental domain (cell at origin)
        origin_idx = len(points) // 2  # Approximate
        if vor.point_region[origin_idx] >= 0:
            region = vor.regions[vor.point_region[origin_idx]]
            if -1 not in region:
                polygon = [vor.vertices[i] for i in region]
                poly = Polygon(polygon, facecolor='yellow', alpha=0.3,
                              edgecolor='red', linewidth=2)
                ax.add_patch(poly)
                
        # Plot lattice points
        ax.scatter(points[:, 0], points[:, 1], s=30, c='red', zorder=3)
        ax.scatter(0, 0, s=100, c='green', marker='*', zorder=4,
                  label='Origin')
                  
        self.set_axis_style(ax, xlabel='$x_1$', ylabel='$x_2$',
                           title='Voronoi Cells (Fundamental Domain)')
        ax.set_aspect('equal')
        
        # Set reasonable limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    # ========== 5. Statistical Diagnostic Plots ==========
    
    def plot_qq_gaussian(self, samples, theoretical_params=None,
                        save_name=None, width='single'):
        """
        Q-Q plot comparing samples to Gaussian.
        
        Args:
            samples: Sample data
            theoretical_params: (mean, std) of theoretical distribution
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        # Standardize samples
        if theoretical_params:
            mean, std = theoretical_params
        else:
            mean, std = np.mean(samples), np.std(samples)
            
        standardized = (samples - mean) / std
        
        # Q-Q plot
        stats.probplot(standardized, dist="norm", plot=ax)
        
        # Style the plot
        ax.get_lines()[0].set_markerfacecolor(self.COLORBLIND_PALETTE[0])
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')
        ax.get_lines()[1].set_linewidth(1.5)
        
        self.set_axis_style(ax, xlabel='Theoretical Quantiles',
                           ylabel='Sample Quantiles',
                           title='Q-Q Plot: Gaussian Check')
                           
        # Add text with statistics
        ks_stat, ks_pval = stats.kstest(standardized, 'norm')
        ax.text(0.05, 0.95, f'KS test p-value: {ks_pval:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
               
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    def plot_tvd_evolution(self, tvd_history, theoretical_bound=None,
                          multiple_chains=False, save_name=None, width='single'):
        """
        Plot TVD evolution over time.
        
        Args:
            tvd_history: TVD values over iterations
            theoretical_bound: Theoretical TVD bound function
            multiple_chains: Whether data contains multiple chains
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        if multiple_chains and isinstance(tvd_history, list):
            # Plot each chain
            for i, chain_tvd in enumerate(tvd_history):
                iterations = chain_tvd['iterations']
                tvd_values = chain_tvd['tvd']
                ax.semilogy(iterations, tvd_values, alpha=0.5,
                           color=self.COLORBLIND_PALETTE[i % len(self.COLORBLIND_PALETTE)],
                           linewidth=1.0)
                           
            # Plot mean
            all_iters = tvd_history[0]['iterations']
            mean_tvd = np.mean([chain['tvd'] for chain in tvd_history], axis=0)
            ax.semilogy(all_iters, mean_tvd, 'k-', linewidth=2.0,
                       label='Mean TVD')
                       
        else:
            # Single chain
            iterations = tvd_history['iterations']
            tvd_values = tvd_history['tvd']
            ax.semilogy(iterations, tvd_values, 'b-', linewidth=1.5,
                       label='Empirical TVD')
                       
            # Confidence intervals if available
            if 'ci_lower' in tvd_history:
                ax.fill_between(iterations, tvd_history['ci_lower'],
                               tvd_history['ci_upper'], alpha=0.3)
                               
        # Theoretical bound
        if theoretical_bound is not None:
            if callable(theoretical_bound):
                t_theory = np.linspace(1, max(iterations), 200)
                tvd_theory = [theoretical_bound(t) for t in t_theory]
            else:
                t_theory = iterations
                tvd_theory = theoretical_bound
                
            ax.semilogy(t_theory, tvd_theory, 'r--', linewidth=1.5,
                       label='Theoretical bound')
                       
        # Reference lines
        ax.axhline(y=0.25, color='green', linestyle=':', linewidth=1.0,
                  alpha=0.7, label=r'$\epsilon = 0.25$')
                  
        self.set_axis_style(ax, xlabel='Iteration $t$',
                           ylabel=r'$\|P^t - \pi\|_{TV}$',
                           title='Total Variation Distance Evolution')
                           
        # Set reasonable y-limits
        ax.set_ylim([1e-4, 2])
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    def plot_importance_weights(self, weights, theoretical_max=None,
                              save_name=None, width='single'):
        """
        Plot distribution of importance weights.
        
        Args:
            weights: Array of importance weights
            theoretical_max: Theoretical maximum weight
            save_name: Filename to save
            width: Figure width
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.get_figure_size(width, 0.5))
        
        # Histogram
        ax1.hist(weights, bins=50, density=True, alpha=0.7,
                color=self.COLORBLIND_PALETTE[0], edgecolor='black')
                
        # Add theoretical max if given
        if theoretical_max:
            ax1.axvline(x=theoretical_max, color='red', linestyle='--',
                       linewidth=1.5, label=f'Theoretical max: {theoretical_max:.2f}')
                       
        # Statistics
        mean_w = np.mean(weights)
        median_w = np.median(weights)
        max_w = np.max(weights)
        
        ax1.axvline(x=mean_w, color='green', linestyle='-', linewidth=1.5,
                   label=f'Mean: {mean_w:.2f}')
        ax1.axvline(x=median_w, color='orange', linestyle='-', linewidth=1.5,
                   label=f'Median: {median_w:.2f}')
                   
        self.set_axis_style(ax1, xlabel='Weight $w(x)$', ylabel='Density',
                           title='Weight Distribution')
                           
        # Log-scale version
        log_weights = np.log(weights[weights > 0])
        ax2.hist(log_weights, bins=50, density=True, alpha=0.7,
                color=self.COLORBLIND_PALETTE[1], edgecolor='black')
                
        self.set_axis_style(ax2, xlabel=r'$\log w(x)$', ylabel='Density',
                           title='Log-Weight Distribution')
                           
        # Add ESS
        ess = len(weights) / (1 + np.var(weights))
        fig.suptitle(f'Importance Weights (ESS = {ess:.1f})', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, (ax1, ax2)
    
    # ========== 6. Algorithm Comparison Plots ==========
    
    def plot_algorithm_comparison(self, results_dict, metrics=['mixing_time',
                                'acceptance_rate', 'spectral_gap'],
                                save_name=None, width='double'):
        """
        Compare multiple algorithms on same axes.
        
        Args:
            results_dict: Results by algorithm
            metrics: Which metrics to compare
            save_name: Filename to save
            width: Figure width
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, 
                                figsize=self.get_figure_size(width, 1.0/n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        algorithms = list(results_dict.keys())
        x_pos = np.arange(len(algorithms))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = []
            errors = []
            
            for algo in algorithms:
                if metric in results_dict[algo]:
                    val = results_dict[algo][metric]['mean']
                    err = results_dict[algo][metric].get('std', 0)
                    values.append(val)
                    errors.append(err)
                else:
                    values.append(0)
                    errors.append(0)
                    
            # Bar plot with error bars
            bars = ax.bar(x_pos, values, yerr=errors, capsize=5,
                          color=[self.COLORBLIND_PALETTE[j % len(self.COLORBLIND_PALETTE)]
                                 for j in range(len(algorithms))],
                          alpha=0.8, edgecolor='black')
                          
            # Styling
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            
            # Metric-specific formatting
            if metric == 'mixing_time':
                ax.set_ylabel('Mixing Time')
                ax.set_yscale('log')
            elif metric == 'acceptance_rate':
                ax.set_ylabel('Acceptance Rate')
                ax.set_ylim([0, 1])
            elif metric == 'spectral_gap':
                ax.set_ylabel('Spectral Gap $\gamma$')
                ax.set_yscale('log')
            else:
                ax.set_ylabel(metric.replace('_', ' ').title())
                
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, axes
    
    def plot_parameter_sensitivity(self, param_values, results_matrix,
                                  param_name='sigma', metric_name='mixing_time',
                                  algorithms=None, save_name=None, width='single'):
        """
        Plot parameter sensitivity analysis.
        
        Args:
            param_values: Parameter values tested
            results_matrix: Results for each parameter/algorithm
            param_name: Name of parameter
            metric_name: Name of metric
            algorithms: Algorithm names
            save_name: Filename to save
            width: Figure width
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size(width))
        
        if algorithms is None:
            algorithms = [f'Algorithm {i}' for i in range(results_matrix.shape[1])]
            
        # Plot each algorithm
        for i, algo in enumerate(algorithms):
            color = self.COLORBLIND_PALETTE[i % len(self.COLORBLIND_PALETTE)]
            
            # Main line
            ax.plot(param_values, results_matrix[:, i], 'o-',
                   color=color, label=algo, linewidth=1.5, markersize=6)
                   
            # Confidence intervals if available
            if results_matrix.shape[2] > 1:
                lower = results_matrix[:, i, 1]
                upper = results_matrix[:, i, 2]
                ax.fill_between(param_values, lower, upper,
                               color=color, alpha=0.2)
                               
        # Formatting
        if param_name == 'sigma':
            ax.set_xlabel(r'$\sigma$')
        else:
            ax.set_xlabel(param_name.replace('_', ' ').title())
            
        if metric_name == 'mixing_time':
            ax.set_ylabel('Mixing Time')
            ax.set_yscale('log')
        elif metric_name == 'acceptance_rate':
            ax.set_ylabel('Acceptance Rate')
            ax.set_ylim([0, 1])
        else:
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            
        self.set_axis_style(ax, title=f'{metric_name.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}')
        
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig, ax
    
    # ========== 7. Export Functions ==========
    
    def save_figure(self, fig, filename, formats=['pdf', 'png']):
        """
        Save figure in multiple formats.
        
        Args:
            fig: Figure object
            filename: Base filename (without extension)
            formats: List of formats to save
        """
        for fmt in formats:
            filepath = os.path.join(self.figure_dir, f'{filename}.{fmt}')
            
            if fmt == 'pdf':
                fig.savefig(filepath, format='pdf', bbox_inches='tight',
                           pad_inches=0.1)
            elif fmt == 'png':
                fig.savefig(filepath, format='png', dpi=self.dpi,
                           bbox_inches='tight', pad_inches=0.1)
            elif fmt == 'svg':
                fig.savefig(filepath, format='svg', bbox_inches='tight',
                           pad_inches=0.1)
                           
        # Also save data for reproducibility
        self._save_figure_data(fig, filename)
        
    def save_for_latex(self, fig, filename):
        """Save figure optimized for LaTeX inclusion."""
        self.save_figure(fig, filename, formats=['pdf'])
        
        # Generate LaTeX snippet
        latex_snippet = f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\columnwidth]{{{filename}.pdf}}
    \\caption{{Caption here}}
    \\label{{fig:{filename}}}
\\end{{figure}}
"""
        
        with open(os.path.join(self.figure_dir, f'{filename}_latex.txt'), 'w') as f:
            f.write(latex_snippet)
            
    def save_for_presentation(self, fig, filename):
        """Save figure optimized for presentations."""
        # Temporarily increase font sizes
        old_size = plt.rcParams['font.size']
        plt.rcParams['font.size'] = 14
        
        self.save_figure(fig, filename + '_presentation', formats=['png'])
        
        # Restore font size
        plt.rcParams['font.size'] = old_size
        
    def generate_tikz(self, data_dict, plot_type='line', filename=None):
        """
        Generate TikZ code for direct LaTeX embedding.
        
        Args:
            data_dict: Dictionary with plot data
            plot_type: Type of plot ('line', 'bar', 'scatter')
            filename: Save TikZ code to file
        """
        tikz_code = """
\\begin{tikzpicture}
\\begin{axis}[
    width=\\columnwidth,
    height=0.6\\columnwidth,
    xlabel={$x$},
    ylabel={$y$},
    grid=major,
    legend pos=north west,
]
"""
        
        if plot_type == 'line':
            for label, data in data_dict.items():
                tikz_code += f"\\addplot coordinates {{\n"
                for x, y in zip(data['x'], data['y']):
                    tikz_code += f"    ({x},{y})\n"
                tikz_code += f"}};\n\\addlegendentry{{{label}}}\n\n"
                
        tikz_code += """
\\end{axis}
\\end{tikzpicture}
"""
        
        if filename:
            filepath = os.path.join(self.figure_dir, f'{filename}.tex')
            with open(filepath, 'w') as f:
                f.write(tikz_code)
                
        return tikz_code
    
    # ========== Helper Methods ==========
    
    def _compute_acf(self, x, lag):
        """Compute autocorrelation at given lag."""
        n = len(x)
        if lag >= n:
            return 0
            
        x_centered = x - np.mean(x)
        c0 = np.dot(x_centered, x_centered) / n
        c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / (n - lag)
        
        return c_lag / c0 if c0 > 0 else 0
    
    def _save_figure_data(self, fig, filename):
        """Save underlying data for reproducibility."""
        data = {}
        
        for i, ax in enumerate(fig.axes):
            ax_data = {
                'lines': [],
                'collections': [],
                'patches': []
            }
            
            # Extract line data
            for line in ax.lines:
                ax_data['lines'].append({
                    'xdata': line.get_xdata().tolist(),
                    'ydata': line.get_ydata().tolist(),
                    'label': line.get_label()
                })
                
            data[f'axis_{i}'] = ax_data
            
        # Save as JSON
        json_path = os.path.join(self.figure_dir, f'{filename}_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def create_figure_template(self, template_type='comparison'):
        """
        Create figure template for common plot types.
        
        Args:
            template_type: Type of template
            
        Returns:
            fig, axes: Figure and axes objects
        """
        if template_type == 'comparison':
            fig, axes = plt.subplots(2, 2, figsize=self.get_figure_size('double', 0.8))
        elif template_type == 'grid':
            fig, axes = plt.subplots(3, 3, figsize=self.get_figure_size('double', 1.0))
        elif template_type == 'side_by_side':
            fig, axes = plt.subplots(1, 2, figsize=self.get_figure_size('double', 0.5))
        else:
            fig, ax = plt.subplots(figsize=self.get_figure_size('single'))
            axes = ax
            
        plt.tight_layout()
        return fig, axes


# Example usage and tests
if __name__ == '__main__':
    # Create plotting tools
    plotter = PlottingTools(style='publication', use_latex=True)
    
    # Test 1: Lattice Gaussian 2D
    n_points = 100
    lattice_points = np.random.randn(n_points, 2) * 3
    lattice_points = np.round(lattice_points)  # Snap to integer lattice
    
    fig, ax = plotter.plot_lattice_gaussian_2d(
        lattice_points, sigma=1.5, save_name='test_lattice_gaussian'
    )
    plt.close()
    
    # Test 2: Convergence comparison
    test_results = {
        'Klein': {
            'tvd_history': {
                'iterations': np.arange(1, 1000, 10),
                'tvd': 0.5 * 0.95**np.arange(1, 1000, 10),
                'ci_lower': 0.4 * 0.95**np.arange(1, 1000, 10),
                'ci_upper': 0.6 * 0.95**np.arange(1, 1000, 10)
            },
            'delta': 0.05
        },
        'MHK': {
            'tvd_history': {
                'iterations': np.arange(1, 1000, 10),
                'tvd': 0.7 * 0.98**np.arange(1, 1000, 10)
            },
            'delta': 0.02
        }
    }
    
    fig, axes = plotter.plot_convergence_comparison(
        test_results, save_name='test_convergence'
    )
    plt.close()
    
    print("Test plots generated successfully!")