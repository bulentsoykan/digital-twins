"""
# ABOUTME: Digital Twins package for data-driven dynamic simulation
# ABOUTME: Provides data assimilation algorithms and simulation models

Digital Twins: Data-Driven Dynamic Simulation Package

This package provides tools for building digital twins with data assimilation
capabilities, including Kalman filters, particle filters, and various simulation models.
"""

__version__ = "0.1.0"

# Import main classes for easier access
from .assimilation.kalman import KalmanFilter, ExtendedKalmanFilter, EnsembleKalmanFilter
from .assimilation.particle import BootstrapParticleFilter, systematic_resampling
from .models.continuous import ContinuousSimulator, ContinuousModel, LorenzSystem
from .models.discrete_time import DiscreteTimeSimulator, DiscreteTimeModel
from .models.devs import DEVSAtomic, DEVSCoordinator

__all__ = [
    # Assimilation algorithms
    "KalmanFilter",
    "ExtendedKalmanFilter", 
    "EnsembleKalmanFilter",
    "BootstrapParticleFilter",
    "systematic_resampling",
    # Models and simulators
    "ContinuousSimulator",
    "ContinuousModel",
    "LorenzSystem",
    "DiscreteTimeSimulator",
    "DiscreteTimeModel",
    "DEVSAtomic",
    "DEVSCoordinator",
]