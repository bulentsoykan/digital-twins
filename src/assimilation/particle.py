"""
src/assimilation/particle.py

Particle Filters (Sequential Monte Carlo).

This module provides the non-parametric, sample-based data assimilation
algorithms capable of handling highly nonlinear, non-Gaussian systems.
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ==========================================
# 1. SYSTEMATIC RESAMPLING (Algorithm 6.3)
# ==========================================

def systematic_resampling(weights: np.ndarray, N: int) -> np.ndarray:
    """
    Algorithm 6.3: Systematic Resampling Algorithm.
    Minimizes Monte Carlo variance by using a single random number
    to select N particles evenly across the cumulative weight distribution.
    
    Args:
        weights: Normalized importance weights of the particles.
        N: Number of particles to resample.
        
    Returns:
        np.ndarray: An array of indices corresponding to the selected particles.
    """
    # Step 1: Construct the cumulative weight sum array
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # Ensure it ends exactly at 1.0 to avoid float rounding errors
    
    indices = np.zeros(N, dtype=int)
    
    # Step 2: Generate a single random number r in[0, 1/N)
    r = np.random.uniform(0, 1.0 / N)
    
    # Symmetrically increase and check with the cumulative weight sum array
    i = 0
    for j in range(N):
        u = r + j / N
        while u > cumulative_sum[i]:
            i += 1
        indices[j] = i
        
    return indices


# ==========================================
# 2. BOOTSTRAP PARTICLE FILTER (Algorithm 6.4)
# ==========================================

class BootstrapParticleFilter:
    """
    The Bootstrap Filter (Sequential Importance Sampling with Resampling).
    Approximates the belief distribution of a state using a set of N particles.
    """
    def __init__(self, N_particles: int, initial_particles: np.ndarray):
        """
        Algorithm 6.4 - Step 1: Initialization
        Args:
            N_particles: The total number of particles (N).
            initial_particles: A pre-sampled (N, state_dim) array representing bel(x_0).
        """
        self.N = N_particles
        self.particles = np.array(initial_particles, copy=True)
        self.state_dim = self.particles.shape[1] if len(self.particles.shape) > 1 else 1
        
        # Initialize weights uniformly
        self.weights = np.ones(self.N) / self.N

    def predict(self, f_func: Callable, process_noise_std: np.ndarray, u: Optional[np.ndarray] = None):
        """
        Algorithm 6.4 - Step 2a: Importance sampling step (Prediction).
        Samples the new state using the state transition model p(x_k | x_{k-1}, u_k).
        
        Args:
            f_func: The state transition function (e.g., a simulation model).
            process_noise_std: Standard deviations for the Gaussian process noise.
            u: External input vector.
        """
        # Apply the deterministic simulation model to each particle
        for i in range(self.N):
            self.particles[i] = f_func(self.particles[i], u)
            
        # Add process noise (Graph noise equivalent for continuous domains)
        noise = np.random.normal(0, process_noise_std, size=(self.N, self.state_dim))
        self.particles += noise
        
        return self.particles

    def rejuvenate(self, rejuvenation_std: np.ndarray):
        """
        Section 7.3.3: Particle Rejuvenation.
        Directly adds stochastic perturbations to particles to prevent sample 
        impoverishment (where all particles collapse onto a single state).
        
        Args:
            rejuvenation_std: Standard deviation of the noise added to each state variable.
        """
        perturbation = np.random.normal(0, rejuvenation_std, size=(self.N, self.state_dim))
        self.particles += perturbation

    def update(self, y_md: float, g_func: Callable, sensor_noise_std: float):
        """
        Algorithm 6.4 - Step 2b & 3: Weight Computation and Resampling.
        Updates the weights based on measurement likelihood, normalizes, and resamples.
        
        Args:
            y_md: The actual scalar measurement data.
            g_func: The measurement model mapping function MF(x_k).
            sensor_noise_std: Standard deviation of the sensor (sigma).
        """
        # 1. Weight Computation (Equation 6.30 / 6.72)
        # Calculate predicted measurement for each particle
        predicted_measurements = np.array([g_func(p) for p in self.particles])
        
        # Calculate likelihood: p(y_k = y^{md} | x_k^{(i)})
        # Using the Gaussian PDF formula
        variance = sensor_noise_std ** 2
        diff = predicted_measurements - y_md
        likelihoods = (1.0 / np.sqrt(2.0 * np.pi * variance)) * np.exp(-0.5 * (diff ** 2) / variance)
        
        # Update weights
        self.weights *= likelihoods
        
        # Prevent division by zero if all particles are highly unlikely
        weight_sum = np.sum(self.weights)
        if weight_sum == 0 or np.isnan(weight_sum):
            self.weights = np.ones(self.N) / self.N
        else:
            # 2. Normalize weights
            self.weights /= weight_sum
            
        # 3. Algorithm 6.4 - Step 3: Resampling Step
        indices = systematic_resampling(self.weights, self.N)
        self.particles = self.particles[indices]
        
        # Reset weights after resampling
        self.weights = np.ones(self.N) / self.N
        
        return self.particles

    def estimate_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the mean and standard deviation of the current particle ensemble."""
        mean_state = np.mean(self.particles, axis=0)
        std_state = np.std(self.particles, axis=0)
        return mean_state, std_state


# ==========================================
# 3. EXECUTABLE DEMONSTRATION (Section 6.8.4)
# ==========================================

if __name__ == "__main__":
    print("Initializing Section 6.8.4: Nonlinear Particle Filter Experiment...")
    print("Scenario: Mobile agent passing two outdoor fireplaces.\n")

    # --- System Parameters ---
    dt = 1.0
    N_PARTICLES = 2000
    
    # State Transition (Eq 6.75)
    # x = [position, velocity]
    def transition_model(x, u=None):
        pos, vel = x[0], x[1]
        new_pos = pos + vel * dt
        new_vel = vel
        return np.array([new_pos, new_vel])
        
    process_noise_std = np.array([1.5, 0.5]) # gamma^p, gamma^v
    
    # Nonlinear Measurement Model (Eqs 6.76 - 6.78)
    T_a = 10.0       # Ambient Temp
    T_c = 30.0       # Temp rise at fire center
    sigma_c = 15.0   # Heat spread constant
    sensor_std = 2.0 # Sensor precision noise
    
    def measurement_model(x):
        pos = x[0]
        # Temperature from Fireplace 1 (x=80)
        T1 = T_c * np.exp(-((pos - 80.0)**2) / (sigma_c**2)) + T_a
        # Temperature from Fireplace 2 (x=200)
        T2 = T_c * np.exp(-((pos - 200.0)**2) / (sigma_c**2)) + T_a
        # Sensor measures the max heat impact
        return max(T1, T2)

    # --- Initialize "True System" ---
    true_pos = 20.0
    true_vel = 10.0
    
    # --- Initialize Particle Filter ---
    # Randomly scatter particles across the 500m road and 0-50m/s speeds
    initial_particles = np.column_stack((
        np.random.uniform(0, 500, N_PARTICLES),
        np.random.uniform(0, 50, N_PARTICLES)
    ))
    
    pf = BootstrapParticleFilter(N_PARTICLES, initial_particles)

    # --- Run Simulation ---
    print(f"{'Step':<5} | {'True Pos':<10} | {'True Vel':<10} | {'Temp Obs':<10} | {'Est Pos':<10} | {'Est Vel':<10}")
    print("-" * 75)

    for k in range(1, 31):
        # 1. True System moves
        true_pos = true_pos + true_vel + np.random.normal(0, 1.5)
        true_vel = true_vel + np.random.normal(0, 0.5)
        
        # 2. Sensor collects noisy data
        true_temp = measurement_model([true_pos, true_vel])
        noisy_measurement = true_temp + np.random.normal(0, sensor_std)
        
        # 3. Particle Filter PREDICT
        pf.predict(transition_model, process_noise_std)
        
        # 4. Particle Filter UPDATE (Resampling happens here)
        pf.update(noisy_measurement, measurement_model, sensor_std)
        
        # Extract mean estimates
        est_state, _ = pf.estimate_state()
        est_pos, est_vel = est_state[0], est_state[1]
        
        if k % 3 == 0 or k == 1:
            print(f"{k:<5} | {true_pos:<10.1f} | {true_vel:<10.1f} | {noisy_measurement:<10.1f} | {est_pos:<10.1f} | {est_vel:<10.1f}")
            
    print("\nResult: Observe how the PF converges rapidly to the true state, overcoming")
    print("the multimodal bimodal ambiguity (left vs right side of the fireplaces) by Step 15.")