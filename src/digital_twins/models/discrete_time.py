"""
src/models/discrete_time.py

Discrete Time Simulation Models.

This module provides the framework for simulating dynamic systems 
that evolve in fixed, discrete time steps (e.g., Cellular Automata, 
simple Agent-Based Models).
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ==========================================
# 1. DISCRETE TIME MODEL BASE CLASS
# ==========================================

class DiscreteTimeModel:
    """
    Abstract base class for a Discrete Time Simulation Model.
    Follows the structure: x(t+1) = \delta(x(t), u(t)), y(t) = \lambda(x(t))
    """
    
    def state_transition(self, t: int, x: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
        """
        The discrete state transition function (\delta).
        Defines the next state: x(t+1).
        Must be overridden by subclasses.
        
        Args:
            t: Current discrete time step (integer)
            x: Current state vector at time t
            u: Current external input vector at time t
            
        Returns:
            np.ndarray: The next state vector for time t + 1
        """
        raise NotImplementedError("Subclasses must implement the state_transition function.")

    def output_function(self, t: int, x: np.ndarray) -> np.ndarray:
        """
        The output function (\lambda).
        Defines y(t) = \lambda(x(t)).
        By default, assumes the output is simply the full state (y = x).
        """
        return x


# ==========================================
# 2. DISCRETE TIME SIMULATOR (Algorithm 3.2)
# ==========================================

class DiscreteTimeSimulator:
    """
    The Simulator that executes a DiscreteTimeModel.
    Implements Algorithm 3.2: Discrete Time Simulation Algorithm.
    """
    
    def __init__(self, model: DiscreteTimeModel):
        """
        Args:
            model: An instance of a DiscreteTimeModel subclass.
        """
        self.model = model
            
    def simulate(
        self, 
        t_start: int, 
        t_end: int, 
        x0: np.ndarray, 
        u_trajectory: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the step-wise simulation loop over a specified time horizon.
        
        Args:
            t_start: Start time step (usually 0)
            t_end: End time step (T_f)
            x0: Initial state vector
            u_trajectory: A function that returns the input vector u at any time step t.
            
        Returns:
            Tuple containing: (time_history, state_history, output_history)
        """
        # Time steps are strictly integers
        num_steps = (t_end - t_start) + 1
        t_history = np.arange(t_start, t_end + 1, dtype=int)
        
        # Pre-allocate state history array
        x_history = np.zeros((num_steps, *x0.shape), dtype=x0.dtype)
        
        # Determine output dimension and pre-allocate
        y0 = self.model.output_function(t_start, x0)
        y_history = np.zeros((num_steps, *np.atleast_1d(y0).shape), dtype=np.atleast_1d(y0).dtype)
        
        # Initialization
        x = np.array(x0, copy=True)
        
        # Simulation Loop (Matches Algorithm 3.2)
        for i, t in enumerate(t_history):
            # 1. Fetch current external input
            u = u_trajectory(t) if u_trajectory else None
            
            # 2. Record output and state ( y(t) = \lambda(x(t)) )
            y_history[i] = self.model.output_function(t, x)
            x_history[i] = x
            
            # 3. Compute next state: x(t+1) = \delta(x(t), u(t))
            if i < num_steps - 1:
                # We use copy to avoid mutating the history arrays inadvertently
                x = self.model.state_transition(t, x, u).copy()
                
        return t_history, x_history, y_history


# ==========================================
# 3. TEXTBOOK EXAMPLE: 1D CELLULAR AUTOMATON
# ==========================================

class CellularAutomaton1D(DiscreteTimeModel):
    """
    1D Cellular Automaton Model (Section 3.4.1).
    Default is Rule 30, which exhibits Class 3 chaotic behavior.
    """
    
    def __init__(self, rule_number: int = 30):
        """
        Args:
            rule_number: Integer between 0 and 255 (Wolfram Code).
        """
        if not (0 <= rule_number <= 255):
            raise ValueError("Rule number must be between 0 and 255.")
            
        self.rule_number = rule_number
        
        # Convert rule number to binary array (e.g., 30 ->[0, 0, 0, 1, 1, 1, 1, 0])
        # Note: Index 0 corresponds to neighborhood '000', Index 7 to '111'
        binary_string = format(rule_number, '08b')[::-1]
        self.rule_array = np.array([int(bit) for bit in binary_string], dtype=int)

    def state_transition(self, t: int, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies the cellular automaton transition rule across the entire grid simultaneously.
        Uses periodic boundary conditions (wrap-around) via np.roll.
        """
        # Shift arrays to represent left, center, and right neighbors
        left = np.roll(x, 1)
        center = x
        right = np.roll(x, -1)
        
        # Calculate neighborhood integer value (0 to 7) for each cell
        # e.g., if left=1, center=0, right=1 -> (4) + (0) + (1) = 5
        neighborhood = (left << 2) | (center << 1) | right
        
        # Map the neighborhood integer to the new state using the rule array
        next_state = self.rule_array[neighborhood]
        
        return next_state


# ==========================================
# 4. EXECUTABLE DEMONSTRATION
# ==========================================

if __name__ == "__main__":
    # Create the model (Rule 30) and simulator
    ca_model = CellularAutomaton1D(rule_number=30)
    simulator = DiscreteTimeSimulator(model=ca_model)
    
    # Initialize a grid of 61 cells, all 0 except for a single 1 in the center
    # This matches the setup required to generate the triangle in Figure 3.6
    grid_size = 61
    initial_state = np.zeros(grid_size, dtype=int)
    initial_state[grid_size // 2] = 1
    
    # Run the simulation for 30 time steps
    t_hist, x_hist, y_hist = simulator.simulate(
        t_start=0, 
        t_end=30, 
        x0=initial_state
    )
    
    # Print the result using ASCII characters to visualize the chaotic growth
    print("Rule 30 Cellular Automaton (Simulation Results):\n")
    for row in x_hist:
        # Convert 1s to '█' and 0s to ' ' (space) for visualization
        visual_row = "".join(['█' if cell == 1 else ' ' for cell in row])
        print(visual_row)