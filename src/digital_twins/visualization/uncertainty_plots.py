"""
src/visualization/uncertainty_plots.py

Visualization tools for Uncertainty, Belief Distributions, and Particles.

This module contains helpers to plot 1D/2D Gaussian distributions 
(for Kalman Filters) and particle histograms/scatters (for Particle Filters).

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats
from typing import Optional, Tuple


# ==========================================
# 1. GAUSSIAN REPRESENTATION (Kalman Filters)
# ==========================================

def plot_1d_gaussian(
    mu: float, 
    variance: float, 
    ax: Optional[plt.Axes] = None, 
    x_range: Optional[Tuple[float, float]] = None,
    fill: bool = True,
    true_state: Optional[float] = None,
    **kwargs
) -> plt.Axes:
    """
    Plots a 1D Gaussian bell curve (PDF).
    Matches textbook Figure 6.1.
    
    Args:
        mu: Mean of the distribution.
        variance: Variance (sigma^2) of the distribution.
        ax: Matplotlib axes. If None, creates a new one.
        x_range: Tuple of (min, max) for the x-axis. Auto-calculated if None.
        fill: Whether to shade the area under the curve.
        true_state: Optional actual state to plot as a vertical marker.
        **kwargs: Matplotlib line styling arguments.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
    sigma = np.sqrt(variance)
    
    if x_range is None:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    else:
        x = np.linspace(x_range[0], x_range[1], 1000)
        
    y = stats.norm.pdf(x, mu, sigma)
    
    line, = ax.plot(x, y, **kwargs)
    
    if fill:
        ax.fill_between(x, y, alpha=0.2, color=line.get_color())
        
    if true_state is not None:
        ax.axvline(true_state, color='black', linestyle=':', label='True State')
        ax.plot(true_state, 0, marker='o', color='black', markersize=6)
        
    ax.set_ylabel("Probability Density")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_2d_covariance_ellipse(
    mu: np.ndarray, 
    Sigma: np.ndarray, 
    ax: Optional[plt.Axes] = None, 
    n_std: float = 2.0, 
    true_state: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Axes:
    """
    Plots a 2D Gaussian Error Ellipse using eigenvalue decomposition.
    Matches textbook Figure 6.2(b).
    
    Args:
        mu: 2D mean vector [x, y].
        Sigma: 2x2 covariance matrix.
        ax: Matplotlib axes.
        n_std: Number of standard deviations for the ellipse radius (2.0 = ~95% confidence).
        true_state: Optional 2D actual state to plot as a marker.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Eigenvalue decomposition to find ellipse axes and rotation
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    
    # Sort eigenvalues/vectors descending
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # Calculate angle and width/height
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    # Defaults for styling
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('edgecolor', 'black')
    kwargs.setdefault('linewidth', 1.5)
    
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    # Plot the mean
    ax.plot(mu[0], mu[1], marker='o', color='black', markersize=4, label='Mean $\mu$')
    
    if true_state is not None:
        ax.plot(true_state[0], true_state[1], marker='x', color='red', markersize=8, label='True State')
        
    ax.set_aspect('equal', 'datalim')
    ax.grid(True, alpha=0.3)
    
    return ax


# ==========================================
# 2. SAMPLE-BASED REPRESENTATION (Particle Filters)
# ==========================================

def plot_1d_particle_histogram(
    particles: np.ndarray, 
    weights: Optional[np.ndarray] = None, 
    ax: Optional[plt.Axes] = None, 
    bins: int = 50, 
    true_state: Optional[float] = None,
    title: str = "Particle Histogram"
) -> plt.Axes:
    """
    Plots a 1D histogram of particles to visualize a sample-based belief.
    Excellent for showing multimodal distributions (Figure 6.20).
    
    Args:
        particles: Array of 1D particle states.
        weights: Optional importance weights. If None, assumes equal weights.
        ax: Matplotlib axes.
        bins: Number of histogram bins.
        true_state: Optional actual state to plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    ax.hist(
        particles, 
        bins=bins, 
        weights=weights, 
        density=True, 
        color='dimgray', 
        edgecolor='black',
        alpha=0.8,
        label='Particle Belief'
    )
    
    if true_state is not None:
        ax.axvline(true_state, color='black', linestyle='--', linewidth=2, label='True State')
        ax.plot(true_state, 0, marker='x', color='black', markersize=10, markeredgewidth=2)
        
    ax.set_title(title)
    ax.set_ylabel("Frequency / Probability Density")
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    return ax


def plot_2d_particles(
    particles: np.ndarray, 
    weights: Optional[np.ndarray] = None, 
    ax: Optional[plt.Axes] = None, 
    true_state: Optional[np.ndarray] = None,
    title: str = "2D Particle Distribution"
) -> plt.Axes:
    """
    Plots a 2D scatter of particles, optionally scaling sizes by their importance weights.
    Matches textbook Figure 6.2(a).
    
    Args:
        particles: Array of shape (N, 2) containing particle states.
        weights: Optional importance weights.
        ax: Matplotlib axes.
        true_state: Optional 2D actual state.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Scale marker sizes based on weights, or use uniform size
    if weights is not None:
        # Normalize weights for visualization purposes so they are visible
        vis_weights = (weights / np.max(weights)) * 100 + 5
    else:
        vis_weights = 20
        
    ax.scatter(
        particles[:, 0], particles[:, 1], 
        s=vis_weights, 
        alpha=0.5, 
        color='steelblue', 
        marker='x',
        label='Particles'
    )
    
    if true_state is not None:
        ax.plot(true_state[0], true_state[1], marker='o', color='red', markersize=6, label='True State')
        
    ax.set_title(title)
    ax.set_aspect('equal', 'datalim')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


# ==========================================
# 3. EXECUTABLE DEMONSTRATION
# ==========================================

if __name__ == "__main__":
    import matplotlib.gridspec as gridspec

    # Set up a nice dashboard figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Digital Twins: Uncertainty & Belief Representation", fontsize=16, fontweight='bold')

    # --- 1. 1D Gaussian (Prior vs Posterior) ---
    ax1 = fig.add_subplot(gs[0, 0])
    # Prior
    plot_1d_gaussian(mu=40.0, variance=25.0, ax=ax1, color='blue', label='Prior $bel(x)$', linestyle='--')
    # Posterior
    plot_1d_gaussian(mu=46.0, variance=5.0, ax=ax1, color='green', label='Posterior $\overline{bel}(x)$', true_state=48.0)
    ax1.set_title("1D Gaussian Representation (Kalman Filter)")
    ax1.legend()

    # --- 2. 1D Particle Histogram (Multimodal) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Simulate a multimodal distribution (e.g., agent doesn't know if it's at fireplace 1 or 2)
    particles_left = np.random.normal(30, 2.0, 500)
    particles_right = np.random.normal(80, 3.0, 500)
    all_particles = np.concatenate([particles_left, particles_right])
    plot_1d_particle_histogram(all_particles, ax=ax2, bins=60, true_state=81.0, title="1D Sample-Based (Multimodal Particle Filter)")

    # --- 3. 2D Particle Scatter ---
    ax3 = fig.add_subplot(gs[1, 0])
    np.random.seed(42)
    p2d = np.random.multivariate_normal([38.0, 54.0], [[2.0, 1.5],[1.5, 3.0]], 200)
    plot_2d_particles(p2d, ax=ax3, true_state=np.array([39.5, 55.0]), title="2D Sample-Based Representation (Fig 6.2a)")
    ax3.set_xlabel("$x_1$ position")
    ax3.set_ylabel("$x_2$ position")

    # --- 4. 2D Covariance Ellipse ---
    ax4 = fig.add_subplot(gs[1, 1])
    mu_2d = np.array([38.0, 54.0])
    Sigma_2d = np.array([[2.0, 1.5], [1.5, 3.0]])
    plot_2d_covariance_ellipse(mu_2d, Sigma_2d, ax=ax4, true_state=np.array([39.5, 55.0]))
    # Overlay the same particles faintly to show equivalence
    ax4.scatter(p2d[:, 0], p2d[:, 1], s=10, alpha=0.2, color='gray')
    ax4.set_title("2D Gaussian Representation (Fig 6.2b)")
    ax4.set_xlabel("$x_1$ position")
    ax4.set_ylabel("$x_2$ position")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()