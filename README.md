# Digital Twins: Data-Driven Dynamic Simulation

[![PyPI version](https://badge.fury.io/py/digital-twins-ddds.svg)](https://badge.fury.io/py/digital-twins-ddds)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Bulent%20Soykan-green)](https://github.com/bulentsoykan)

## Overview

Digital Twins is a Python package providing state-of-the-art algorithms for building digital twins through data assimilation. It combines simulation models with real-world observations to create accurate, real-time representations of physical systems.

### What is a Digital Twin?

A digital twin is a virtual representation of a physical system that updates in real-time using sensor data. This package provides the mathematical foundation and algorithms needed to:

- **Simulate** dynamic systems using continuous, discrete-time, and event-based models
- **Assimilate** real-world data to correct model predictions
- **Estimate** hidden states and parameters from noisy observations
- **Visualize** uncertainty and spatial distributions

## Key Features

### 🎯 Data Assimilation Algorithms
- **Kalman Filters**: Standard, Extended (EKF), and Ensemble (EnKF) implementations
- **Particle Filters**: Bootstrap filter with systematic resampling for nonlinear/non-Gaussian systems

### 🔬 Simulation Models
- **Continuous Models**: ODE solvers (Euler, RK4) for systems like the Lorenz attractor
- **Discrete-Time Models**: Including cellular automata and difference equations
- **DEVS Models**: Discrete Event System Specification for event-driven simulations

### 📊 Visualization Tools
- Uncertainty visualization (confidence ellipses, particle clouds)
- Spatial distribution plots for grid-based simulations
- Real-time animation support via Jupyter widgets

## Installation

### From PyPI (Recommended)

```bash
pip install digital-twins-dds
```

### From Source

```bash
git clone https://github.com/bulentsoykan/digital-twins.git
cd digital-twins
pip install -e .
```

### Development Installation

```bash
pip install -e .[dev]  # Includes testing and development tools
```

## Quick Start

### Example 1: Kalman Filter for State Estimation

```python
import numpy as np
from digital_twins import KalmanFilter

# Initialize with uncertain initial state
mu0 = np.array([0.0, 0.0])  # [position, velocity]
Sigma0 = np.eye(2) * 100    # High initial uncertainty

kf = KalmanFilter(mu0, Sigma0)

# System dynamics (constant velocity model)
dt = 1.0
A = np.array([[1.0, dt], [0.0, 1.0]])  # State transition
Q = np.eye(2) * 0.1                     # Process noise
C = np.array([[1.0, 0.0]])              # Observe position only
R = np.array([[1.0]])                   # Measurement noise

# Simulate one step
kf.predict(A, Q)                        # Predict next state
measurement = np.array([5.0])           # Receive sensor data
kf.update(measurement, C, R)            # Correct with measurement

print(f"Estimated state: {kf.mu}")
print(f"Uncertainty: {kf.Sigma}")
```

### Example 2: Particle Filter for Nonlinear Systems

```python
from digital_twins import BootstrapParticleFilter

# Initialize 1000 particles
N_particles = 1000
initial_particles = np.random.randn(N_particles, 2)  # Random initial distribution

pf = BootstrapParticleFilter(N_particles, initial_particles)

# Nonlinear dynamics
def f(x, u):
    return np.array([x[0] + 0.1*np.sin(x[1]), x[1] + 0.1])

def g(x):
    return x[0]**2  # Nonlinear measurement

# Run filter
pf.predict(f, process_noise_std=np.array([0.1, 0.1]))
pf.update(y_md=2.5, g_func=g, sensor_noise_std=0.5)

mean_state, std_state = pf.estimate_state()
print(f"Estimated state: {mean_state} ± {std_state}")
```

### Example 3: Continuous System Simulation

```python
from digital_twins import ContinuousSimulator, LorenzSystem

# Create Lorenz attractor model
model = LorenzSystem(sigma=10.0, rho=28.0, beta=8.0/3.0)
simulator = ContinuousSimulator(model, method='rk4')

# Run simulation
initial_state = np.array([1.0, 1.0, 1.0])
t_history, x_history, y_history = simulator.simulate(
    t_start=0.0, 
    t_end=50.0, 
    dt=0.01, 
    x0=initial_state
)

# x_history contains the chaotic trajectory
```

## Interactive Notebooks

The package includes comprehensive Jupyter notebooks demonstrating:

1. **Continuous Systems**: Lorenz attractor visualization
2. **Discrete-Time Models**: Cellular automata patterns
3. **DEVS Models**: Traffic light coordination
4. **Kalman Filters**: Interactive gain visualization
5. **Particle Filters**: Bootstrap filter basics
6. **Advanced Topics**: Joint state-parameter estimation, spatial tracking

### Running the Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Navigate to notebooks directory
cd notebooks/

# Start Jupyter
jupyter notebook
```

## Package Structure

```
digital-twins/
├── src/digital_twins/
│   ├── assimilation/      # Data assimilation algorithms
│   │   ├── kalman.py      # Kalman filter variants
│   │   └── particle.py    # Particle filters
│   ├── models/            # Simulation models
│   │   ├── continuous.py  # ODE-based models
│   │   ├── discrete_time.py # Difference equations
│   │   └── devs.py        # Event-driven models
│   └── visualization/     # Plotting utilities
├── notebooks/             # Interactive tutorials
├── tests/                 # Unit tests
└── data/                  # Example datasets
```

## Mathematical Background

This package implements algorithms from modern data assimilation theory:

- **Kalman Filtering**: Optimal state estimation for linear Gaussian systems
- **Extended Kalman Filter**: First-order linearization for nonlinear systems  
- **Ensemble Kalman Filter**: Sample-based covariance estimation
- **Particle Filtering**: Sequential Monte Carlo for arbitrary distributions

The mathematical formulations follow standard texts in data assimilation and state estimation.

## Use Cases

Digital twins built with this package can be applied to:

- 🚗 **Transportation**: Traffic flow estimation and prediction
- 🔥 **Wildfire Tracking**: Real-time fire spread monitoring
- 🏭 **Manufacturing**: Production line optimization
- 🌊 **Environmental Monitoring**: Ocean/atmosphere modeling
- 📦 **Supply Chain**: Inventory and logistics optimization
- 🤖 **Robotics**: Sensor fusion and SLAM

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/bulentsoykan/digital-twins) for:

- Bug reports and feature requests
- Pull request guidelines
- Development setup instructions

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=digital_twins
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{digital_twins,
  author = {Soykan, Bulent},
  title = {Digital Twins: Data-Driven Dynamic Simulation},
  year = {2026},
  url = {https://github.com/bulentsoykan/digital-twins}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Author

**Bulent Soykan**
- Website: [bulentsoykan.com](https://bulentsoykan.com)
- GitHub: [@bulentsoykan](https://github.com/bulentsoykan)
- LinkedIn: [Bulent Soykan](https://www.linkedin.com/in/bulent-soykan/)

## Acknowledgments

This package was developed as a companion to research in digital twin technology and data assimilation methods. Special thanks to the scientific computing and data assimilation communities for their foundational work.
