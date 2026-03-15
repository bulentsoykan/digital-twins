"""
src/models/continuous.py

Continuous Simulation Models and Differential Equation Solvers.
Companion code for Chapter 3: Simulation Models and Algorithms.

This module provides the mathematical integrators (Euler, RK4) to 
simulate continuous-time dynamic systems defined by ordinary 
differential equations (ODEs).
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ==========================================
# 1. INTEGRATORS (SOLVERS)
# ==========================================

def euler_step(
    derivative_func: Callable, 
    t: float, 
    x: np.ndarray, 
    u: Optional[np.ndarray], 
    dt: float
) -> np.ndarray:
    """
    Performs a single step of the Euler integration method.
    Matches Equation (3.4) from the textbook.
    
    Args:
        derivative_func: Function representing dx/dt = f(t, x, u)
        t: Current time
        x: Current state vector
        u: Current external input vector (can be None)
        dt: Time step interval
        
    Returns:
        np.ndarray: The updated state vector for time t + dt
    """
    dx_dt = derivative_func(t, x, u)
    return x + dt * dx_dt


def rk4_step(
    derivative_func: Callable, 
    t: float, 
    x: np.ndarray, 
    u: Optional[np.ndarray], 
    dt: float
) -> np.ndarray:
    """
    Performs a single step of the 4th-Order Runge-Kutta (RK4) integration method.
    RK4 offers a much better speed/accuracy trade-off than the Euler method.
    
    Args:
        derivative_func: Function representing dx/dt = f(t, x, u)
        t: Current time
        x: Current state vector
        u: Current external input vector (can be None)
        dt: Time step interval
        
    Returns:
        np.ndarray: The updated state vector for time t + dt
    """
    k1 = derivative_func(t, x, u)
    k2 = derivative_func(t + dt / 2.0, x + (dt / 2.0) * k1, u)
    k3 = derivative_func(t + dt / 2.0, x + (dt / 2.0) * k2, u)
    k4 = derivative_func(t + dt, x + dt * k3, u)
    
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ==========================================
# 2. CONTINUOUS MODEL BASE CLASS
# ==========================================

class ContinuousModel:
    """
    Abstract base class for a Continuous Simulation Model.
    Follows the structure: Model = < u, y, x, \delta_t, \lambda_t >
    """
    
    def state_transition(self, t: float, x: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
        """
        The continuous state transition function (\delta_t).
        Defines the ODE: dx/dt = f(t, x, u).
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the state_transition function.")

    def output_function(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        The output function (\lambda_t).
        Defines y(t) = g(x(t)).
        By default, assumes the output is simply the full state (y = x).
        """
        return x


# ==========================================
# 3. CONTINUOUS SIMULATOR 
# ==========================================

class ContinuousSimulator:
    """
    The Simulator that executes a ContinuousModel.
    Separating the model from the simulator matches the framework in Fig 3.4.
    """
    
    def __init__(self, model: ContinuousModel, method: str = 'rk4'):
        """
        Args:
            model: An instance of a ContinuousModel subclass.
            method: 'euler' or 'rk4'
        """
        self.model = model
        self.method = method.lower()
        if self.method not in ['euler', 'rk4']:
            raise ValueError("Method must be 'euler' or 'rk4'.")
            
    def simulate(
        self, 
        t_start: float, 
        t_end: float, 
        dt: float, 
        x0: np.ndarray, 
        u_trajectory: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Algorithm 3.1: Continuous Simulation Algorithm.
        Runs the simulation loop over a specified time horizon.
        
        Args:
            t_start: Start time
            t_end: End time (T_f)
            dt: Time interval (\Delta t)
            x0: Initial state vector
            u_trajectory: A function that returns the input vector u at any time t.
            
        Returns:
            Tuple containing: (time_history, state_history, output_history)
        """
        # Pre-allocate arrays for performance
        num_steps = int(np.ceil((t_end - t_start) / dt)) + 1
        t_history = np.linspace(t_start, t_end, num_steps)
        x_history = np.zeros((num_steps, len(x0)))
        
        # Determine output dimension
        y0 = self.model.output_function(t_start, x0)
        y_history = np.zeros((num_steps, len(np.atleast_1d(y0))))
        
        # Initialization
        x = np.array(x0, dtype=float)
        
        # Simulation Loop
        for i, t in enumerate(t_history):
            # 1. Fetch current external input
            u = u_trajectory(t) if u_trajectory else None
            
            # 2. Record state and compute output
            x_history[i] = x
            y_history[i] = self.model.output_function(t, x)
            
            # 3. Compute next state (except on the very last step)
            if i < num_steps - 1:
                if self.method == 'euler':
                    x = euler_step(self.model.state_transition, t, x, u, dt)
                elif self.method == 'rk4':
                    x = rk4_step(self.model.state_transition, t, x, u, dt)
                    
        return t_history, x_history, y_history


# ==========================================
# 4. EXAMPLE: THE LORENZ SYSTEM
# ==========================================

class LorenzSystem(ContinuousModel):
    """
    The Lorenz System (Section 3.3.1, Equation 3.5).
    A classic example of continuous simulation modeling chaotic behavior.
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def state_transition(self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluates the Lorenz equations.
        x[0] = X, x[1] = Y, x[2] = Z
        """
        X_val, Y_val, Z_val = x[0], x[1], x[2]
        
        dX_dt = self.sigma * (Y_val - X_val)
        dY_dt = X_val * (self.rho - Z_val) - Y_val
        dZ_dt = (X_val * Y_val) - (self.beta * Z_val)
        
        return np.array([dX_dt, dY_dt, dZ_dt])

# Example usage for a standalone script test:
if __name__ == "__main__":
    # Create the model and simulator
    lorenz_model = LorenzSystem()
    simulator = ContinuousSimulator(model=lorenz_model, method='rk4')
    
    # Initial state exactly as defined in Section 3.3.1
    initial_state = np.array([1.0, 1.0, 1.0])
    
    # Run the simulation
    t_hist, x_hist, y_hist = simulator.simulate(
        t_start=0.0, 
        t_end=40.0, 
        dt=0.01, 
        x0=initial_state
    )
    
    print(f"Simulation completed successfully.")
    print(f"Final Time: {t_hist[-1]:.2f}")
    print(f"Final State (X, Y, Z): {x_hist[-1]}")
