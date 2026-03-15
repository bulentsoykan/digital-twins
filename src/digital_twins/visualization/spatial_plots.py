"""
src/visualization/spatial_plots.py

Visualization tools for Spatiotemporal Systems and Cellular Automata.

This module contains helpers to plot 2D discrete grids (Wildfire states, 
Traffic/Cellular Automata) and continuous 2D heatmaps (Temperature fields, 
Sensor networks). 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Optional, Tuple, List


# ==========================================
# 1. DISCRETE SPATIAL GRIDS (Cellular Automata)
# ==========================================

def plot_discrete_grid(
    grid: np.ndarray,
    cmap: ListedColormap,
    ax: Optional[plt.Axes] = None,
    title: str = "Discrete Spatial Grid",
    show_gridlines: bool = False
) -> plt.Axes:
    """
    Generic plotter for 2D discrete cellular spaces (e.g., CA, Traffic).
    
    Args:
        grid: 2D numpy array of integer states.
        cmap: Matplotlib ListedColormap defining the state colors.
        ax: Matplotlib axes. If None, creates a new one.
        title: Title of the plot.
        show_gridlines: Whether to draw grid lines between cells.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Create boundaries for the discrete color mapping
    n_colors = cmap.N
    bounds = np.arange(-0.5, n_colors + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    cax = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
    
    ax.set_title(title)
    
    if show_gridlines:
        # Minor ticks dictate the grid lines
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
    
    # Hide major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax


def plot_wildfire_state(
    fire_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Wildfire Spread State"
) -> plt.Axes:
    """
    Plots the discrete state of a wildfire simulation.
    State mapping: 0 = Unburned (Green), 1 = Burning (Red), 2 = Burned (Black).
    
    Args:
        fire_grid: 2D numpy array containing values 0, 1, or 2.
        ax: Matplotlib axes.
        title: Title of the plot.
    """
    # Colors exactly as described in Section 8.3 (Fig 8.4)
    # Green = unburned fuel, Red = active fire front, Black = burned out
    fire_cmap = ListedColormap(['#8FBC8F', '#FF0000', '#000000'])
    
    return plot_discrete_grid(fire_grid, cmap=fire_cmap, ax=ax, title=title, show_gridlines=False)


# ==========================================
# 2. CONTINUOUS 2D HEATMAPS (Sensors & Likelihood)
# ==========================================

def plot_continuous_heatmap(
    field: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'hot',
    sensor_locations: Optional[np.ndarray] = None,
    title: str = "Continuous Heatmap"
) -> plt.Axes:
    """
    Plots a continuous 2D heatmap (e.g., temperature field, likelihood map).
    Matches textbook Figure 8.5(a) - (d).
    
    Args:
        field: 2D numpy array of continuous values (e.g., temperatures).
        ax: Matplotlib axes.
        cmap: String name of the matplotlib colormap. 'hot' matches Fig 8.5.
        sensor_locations: Optional (N, 2) array of [y, x] indices for sensors.
        title: Title of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # The 'hot' colormap seamlessly transitions from black->red->yellow->white
    im = ax.imshow(field, cmap=cmap, interpolation='bilinear')
    
    # Overlay sensor locations if provided
    if sensor_locations is not None:
        # Scatter takes (x, y), so we pass (col, row)
        ax.scatter(
            sensor_locations[:, 1], 
            sensor_locations[:, 0], 
            c='cyan', 
            s=15, 
            edgecolor='black', 
            label='Sensors',
            zorder=5
        )
        ax.legend(loc='upper right')
        
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax


# ==========================================
# 3. EXECUTABLE DEMONSTRATION
# ==========================================

if __name__ == "__main__":
    import matplotlib.gridspec as gridspec

    # Set up a nice dashboard figure
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Digital Twins: Spatiotemporal Visualization", fontsize=18, fontweight='bold')

    # ---------------------------------------------------------
    # 1. 1D Cellular Automata History 
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Generate a mock Sierpinski-like triangle (Rule 90/30 approximation)
    size = 50
    ca_history = np.zeros((size, size * 2 + 1), dtype=int)
    ca_history[0, size] = 1
    for i in range(1, size):
        for j in range(1, size * 2):
            # Simple XOR rule for visual effect
            ca_history[i, j] = ca_history[i-1, j-1] ^ ca_history[i-1, j+1]
            
    bw_cmap = ListedColormap(['white', 'black'])
    plot_discrete_grid(ca_history, cmap=bw_cmap, ax=ax1, title="1D Cellular Automata History (Fig 3.6)")
    ax1.set_ylabel("Time Step (Generation)")
    ax1.set_xlabel("1D Spatial Cell Index")

    # ---------------------------------------------------------
    # 2. Sheep-Grass Ecosystem CA (Fig 3.11)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 0 = Empty (White), 1 = Grass (Green), 2 = Sheep (Red)
    eco_grid = np.random.choice([0, 1, 2], size=(40, 40), p=[0.6, 0.3, 0.1])
    eco_cmap = ListedColormap(['#FFFFFF', '#32CD32', '#FF0000'])
    plot_discrete_grid(eco_grid, cmap=eco_cmap, ax=ax2, title="Sheep-Grass Ecosystem CA (Fig 3.11)", show_gridlines=True)

    # ---------------------------------------------------------
    # 3. Wildfire Discrete State 
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Generate a mock fire spread grid (50x50)
    grid_size = 50
    y, x = np.ogrid[-25:25, -25:25]
    distance = np.sqrt(x**2 + y**2)
    
    fire_state = np.zeros((grid_size, grid_size), dtype=int)
    # Burned area (inner core)
    fire_state[distance < 10] = 2
    # Burning area (active front ring)
    fire_state[(distance >= 10) & (distance < 14)] = 1
    # Add some ragged edges to the fire front
    noise = np.random.normal(0, 2, (grid_size, grid_size))
    fire_state[(distance + noise >= 14) & (distance + noise < 16)] = 1
    
    plot_wildfire_state(fire_state, ax=ax3)

    # ---------------------------------------------------------
    # 4. Wildfire Continuous Temperature Heatmap & Sensors 
    # ---------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Generate a temperature field based on the fire state 
    T_a = 27.0 # Ambient
    T_c = 376.0 # Fire temp
    
    # Create a smooth heat gradient spreading out from the center
    temperature_field = T_c * np.exp(-(distance**2) / (2 * 8.0**2)) + T_a
    
    # Add random deployment sensors (Random Deployment Schema 1)
    n_sensors = 40
    sensor_y = np.random.randint(5, 45, n_sensors)
    sensor_x = np.random.randint(5, 45, n_sensors)
    sensors = np.column_stack((sensor_y, sensor_x))
    
    plot_continuous_heatmap(
        temperature_field, 
        ax=ax4, 
        cmap='hot', 
        sensor_locations=sensors, 
        title="Real-Time Temp Measurement Data "
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
