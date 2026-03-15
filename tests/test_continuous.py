"""
# ABOUTME: Tests for continuous simulation models and integrators
# ABOUTME: Validates Euler and RK4 methods against analytical solutions

Test continuous simulation models and integrators.
"""

import numpy as np
import pytest
from digital_twins.models.continuous import euler_step, rk4_step, ContinuousModel, ContinuousSimulator


class SimpleLinearODE(ContinuousModel):
    """Test model: dx/dt = -x with analytical solution x(t) = x0 * exp(-t)"""
    
    def state_transition(self, t, x, u=None):
        return -x


class SimpleHarmonicOscillator(ContinuousModel):
    """Test model: d2x/dt2 = -x (as system: dx/dt = v, dv/dt = -x)"""
    
    def state_transition(self, t, x, u=None):
        # x[0] is position, x[1] is velocity  
        return np.array([x[1], -x[0]])


def test_euler_step_simple_decay():
    """Test Euler method on exponential decay ODE."""
    # dx/dt = -x, x(0) = 1
    def derivative(t, x, u):
        return -x
    
    x0 = np.array([1.0])
    dt = 0.01
    t = 0.0
    
    # Single Euler step
    x1 = euler_step(derivative, t, x0, None, dt)
    
    # Expected: x1 ≈ x0 * (1 - dt) for small dt
    expected = x0 * (1 - dt)
    assert np.allclose(x1, expected, rtol=1e-10)


def test_rk4_step_simple_decay():
    """Test RK4 method on exponential decay ODE."""
    # dx/dt = -x, x(0) = 1
    def derivative(t, x, u):
        return -x
    
    x0 = np.array([1.0])
    dt = 0.1
    t = 0.0
    
    # Single RK4 step
    x1 = rk4_step(derivative, t, x0, None, dt)
    
    # Analytical solution: x(dt) = exp(-dt)
    analytical = np.exp(-dt)
    
    # RK4 should be much more accurate than Euler
    assert np.abs(x1[0] - analytical) < 1e-5


def test_euler_vs_rk4_accuracy():
    """Compare Euler and RK4 accuracy on a known problem."""
    model = SimpleLinearODE()
    x0 = np.array([1.0])
    t_end = 1.0
    dt = 0.1
    
    # Simulate with Euler
    sim_euler = ContinuousSimulator(model, method='euler')
    t_hist_e, x_hist_e, _ = sim_euler.simulate(0.0, t_end, dt, x0)
    
    # Simulate with RK4
    sim_rk4 = ContinuousSimulator(model, method='rk4')
    t_hist_r, x_hist_r, _ = sim_rk4.simulate(0.0, t_end, dt, x0)
    
    # Analytical solution at t_end
    analytical_final = np.exp(-t_end)
    
    # RK4 should be more accurate than Euler
    euler_error = np.abs(x_hist_e[-1, 0] - analytical_final)
    rk4_error = np.abs(x_hist_r[-1, 0] - analytical_final)
    
    assert rk4_error < euler_error
    assert rk4_error < 1e-4


def test_harmonic_oscillator_energy_conservation():
    """Test that RK4 approximately conserves energy in harmonic oscillator."""
    model = SimpleHarmonicOscillator()
    x0 = np.array([1.0, 0.0])  # Start at x=1, v=0
    
    sim = ContinuousSimulator(model, method='rk4')
    t_hist, x_hist, _ = sim.simulate(0.0, 10.0, 0.01, x0)
    
    # Energy = 0.5 * (x^2 + v^2) should be approximately constant
    energy = 0.5 * (x_hist[:, 0]**2 + x_hist[:, 1]**2)
    
    # Check energy conservation (should be close to initial energy)
    initial_energy = 0.5 * (x0[0]**2 + x0[1]**2)
    energy_variation = np.abs(energy - initial_energy)
    
    # Energy should be conserved to within 1%
    assert np.max(energy_variation) < 0.01 * initial_energy


def test_continuous_simulator_with_input():
    """Test simulator with external input."""
    class ControlledSystem(ContinuousModel):
        def state_transition(self, t, x, u):
            # dx/dt = -x + u
            return -x + (u if u is not None else 0)
    
    model = ControlledSystem()
    x0 = np.array([0.0])
    
    # Constant input u = 1
    def u_trajectory(t):
        return np.array([1.0])
    
    sim = ContinuousSimulator(model, method='rk4')
    t_hist, x_hist, _ = sim.simulate(0.0, 5.0, 0.01, x0, u_trajectory)
    
    # System should converge to steady state where dx/dt = 0, so x = u = 1
    assert np.abs(x_hist[-1, 0] - 1.0) < 0.01


def test_invalid_method_raises_error():
    """Test that invalid integration method raises error."""
    model = SimpleLinearODE()
    with pytest.raises(ValueError):
        ContinuousSimulator(model, method='invalid')


def test_output_function():
    """Test custom output function."""
    class SystemWithOutput(ContinuousModel):
        def state_transition(self, t, x, u=None):
            return -x
        
        def output_function(self, t, x):
            # Output is square of state
            return x**2
    
    model = SystemWithOutput()
    x0 = np.array([2.0])
    
    sim = ContinuousSimulator(model, method='euler')
    t_hist, x_hist, y_hist = sim.simulate(0.0, 1.0, 0.1, x0)
    
    # Check that output is square of state
    expected_output = x_hist**2
    assert np.allclose(y_hist, expected_output)