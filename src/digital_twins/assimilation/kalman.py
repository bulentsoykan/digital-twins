"""
src/assimilation/kalman.py

Kalman Filter and its Extensions (EKF, EnKF).

This module provides the Gaussian-representation data assimilation
algorithms for filtering dynamic states over time using noisy measurements.
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ==========================================
# 1. STANDARD KALMAN FILTER (Algorithm 6.2)
# ==========================================

class KalmanFilter:
    """
    Standard Kalman Filter for Linear Gaussian state-space models.
    Matches Equations (6.37) - (6.43) and Algorithm 6.2.
    """
    def __init__(self, mu0: np.ndarray, Sigma0: np.ndarray):
        """
        Initialization (k = 0).
        Args:
            mu0: Initial state mean vector (n,)
            Sigma0: Initial state covariance matrix (n, n)
        """
        self.mu = np.array(mu0, dtype=float)
        self.Sigma = np.array(Sigma0, dtype=float)
        self.n = len(self.mu)

    def predict(self, A: np.ndarray, Q: np.ndarray, B: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction Step (Equations 6.39 & 6.40).
        Computes the prior distribution bel(x_k).
        """
        # \bar{\mu}_k = A_k \mu_{k-1} + B_k u_k
        self.mu = A @ self.mu
        if B is not None and u is not None:
            self.mu += B @ u
            
        # \bar{\Sigma}_k = A_k \Sigma_{k-1} A_k^T + Q_k
        self.Sigma = A @ self.Sigma @ A.T + Q
        
        return self.mu, self.Sigma

    def update(self, y_md: np.ndarray, C: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measurement Update Step (Equations 6.41 - 6.43).
        Computes the posterior distribution bel(x_k).
        """
        # Innovation Covariance: S_k = C_k \bar{\Sigma}_k C_k^T + R_k
        S = C @ self.Sigma @ C.T + R
        
        # Kalman Gain: K_k = \bar{\Sigma}_k C_k^T S_k^{-1}  (Eq 6.43)
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        
        # Posterior Mean: \mu_k = \bar{\mu}_k + K_k (y_k^{md} - C_k \bar{\mu}_k)  (Eq 6.41)
        innovation = y_md - (C @ self.mu)
        self.mu = self.mu + K @ innovation
        
        # Posterior Covariance: \Sigma_k = (I - K_k C_k) \bar{\Sigma}_k  (Eq 6.42)
        I = np.eye(self.n)
        self.Sigma = (I - K @ C) @ self.Sigma
        
        return self.mu, self.Sigma


# ==========================================
# 2. EXTENDED KALMAN FILTER (EKF)
# ==========================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for Nonlinear models with Gaussian noises.
    Matches Section 6.7.3.1 (Equations 6.49 - 6.54).
    """
    def __init__(self, mu0: np.ndarray, Sigma0: np.ndarray):
        self.mu = np.array(mu0, dtype=float)
        self.Sigma = np.array(Sigma0, dtype=float)
        self.n = len(self.mu)

    def predict(self, f_func: Callable, F_jacobian: np.ndarray, Q: np.ndarray, u: Optional[np.ndarray] = None):
        """
        Prediction Step for EKF.
        Args:
            f_func: Nonlinear state transition function f(x, u)
            F_jacobian: Evaluated Jacobian matrix F_k (Eq 6.54)
            Q: Process noise covariance
            u: External input
        """
        # Eq 6.49
        self.mu = f_func(self.mu, u)
        # Eq 6.50
        self.Sigma = F_jacobian @ self.Sigma @ F_jacobian.T + Q
        return self.mu, self.Sigma

    def update(self, y_md: np.ndarray, g_func: Callable, G_jacobian: np.ndarray, R: np.ndarray):
        """
        Measurement Update Step for EKF.
        Args:
            y_md: Actual measurement data
            g_func: Nonlinear measurement function g(x)
            G_jacobian: Evaluated Jacobian matrix G_k (Eq 6.54)
            R: Measurement noise covariance
        """
        # Eq 6.51 (Kalman Gain)
        S = G_jacobian @ self.Sigma @ G_jacobian.T + R
        K = self.Sigma @ G_jacobian.T @ np.linalg.inv(S)
        
        # Eq 6.52 (Posterior Mean)
        innovation = y_md - g_func(self.mu)
        self.mu = self.mu + K @ innovation
        
        # Eq 6.53 (Posterior Covariance)
        I = np.eye(self.n)
        self.Sigma = (I - K @ G_jacobian) @ self.Sigma
        
        return self.mu, self.Sigma


# ==========================================
# 3. ENSEMBLE KALMAN FILTER (EnKF)
# ==========================================

class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter for Nonlinear models. 
    Represents belief as an ensemble of samples rather than a strict Gaussian covariance.
    Matches Section 6.7.3.2 (Equations 6.55 - 6.62).
    """
    def __init__(self, N_particles: int, mu0: np.ndarray, Sigma0: np.ndarray):
        self.N = N_particles
        self.n = len(mu0)
        # Initialize ensemble by sampling from initial prior Gaussian
        self.ensemble = np.random.multivariate_normal(mu0, Sigma0, size=self.N)

    def predict(self, f_func: Callable, Q: np.ndarray, u: Optional[np.ndarray] = None):
        """
        Prediction Step for EnKF. Applies nonlinear model to every particle.
        Matches Equation 6.55.
        """
        process_noise = np.random.multivariate_normal(np.zeros(self.n), Q, size=self.N)
        
        for i in range(self.N):
            self.ensemble[i] = f_func(self.ensemble[i], u) + process_noise[i]
            
        return self.ensemble

    def update(self, y_md: np.ndarray, g_func: Callable, R: np.ndarray):
        """
        Measurement Update Step for EnKF.
        Approximates covariances directly from the sample spread.
        """
        m = len(y_md)
        
        # 1. Compute predicted measurements for each particle without noise: g(\bar{x}_k^{(i)})
        g_X = np.array([g_func(x) for x in self.ensemble])  # Shape: (N, m)
        # Ensure g_X has the right shape even for scalar measurements
        if g_X.ndim == 1:
            g_X = g_X.reshape(-1, 1)
        
        # 2. Compute means (Eqs 6.61, 6.62)
        x_mean = np.mean(self.ensemble, axis=0)             # Shape: (n,)
        g_mean = np.mean(g_X, axis=0)                       # Shape: (m,)
        
        # 3. Compute Deviations
        dX = self.ensemble - x_mean                         # Shape: (N, n)
        dg = g_X - g_mean                                   # Shape: (N, m)
        
        # 4. Approximate Cross-Covariance and Measurement Covariance (Eqs 6.59, 6.60)
        # Vectorized implementation of 1/(N-1) \sum (...)
        Sigma_G_T = (dX.T @ dg) / (self.N - 1)              # Shape: (n, m)
        G_Sigma_G_T = (dg.T @ dg) / (self.N - 1)            # Shape: (m, m)
        
        # 5. Calculate Kalman Gain (Eq 6.58)
        K = Sigma_G_T @ np.linalg.inv(G_Sigma_G_T + R)      # Shape: (n, m)
        
        # 6. Generate virtual noisy measurements for each particle (Eq 6.57)
        measurement_noise = np.random.multivariate_normal(np.zeros(m), R, size=self.N)
        Y_tilde = g_X + measurement_noise                   # Shape: (N, m)
        
        # 7. Shift particles toward the real measurement (Eq 6.56)
        innovation = y_md - Y_tilde                         # Shape: (N, m)
        # Handle 1D case for ensemble update
        if self.n == 1 and K.shape == (1, 1):
            self.ensemble = self.ensemble.reshape(-1, 1) + innovation @ K.T
        else:
            self.ensemble = self.ensemble + innovation @ K.T    # Shape: (N, n)
        
        return self.ensemble


# ==========================================
# 4. EXECUTABLE DEMONSTRATION: 1D AGENT (Sec 6.7.4)
# ==========================================

if __name__ == "__main__":
    print("Initializing Section 6.7.4: 1D Mobile Agent KF Experiment...")
    
    # Time step
    dt = 1.0 
    
    # State: [position, velocity]. Initialized with massive uncertainty.
    mu0 = np.array([0.0, 0.0])
    Sigma0 = np.array([[1000**2, 0.0],[0.0, 1000**2]])
    
    # State Transition Matrix (A)
    # x_p(k) = x_p(k-1) + dt * x_v(k-1)
    # x_v(k) = x_v(k-1)
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
                  
    # Observation Matrix (C) -> We only observe position
    C = np.array([[1.0, 0.0]])
    
    # Process Noise Covariance (Q) -> Pos noise std=2.0, Vel noise std=0.5
    Q = np.array([[2.0**2, 0.0],
                  [0.0, 0.5**2]])
                  
    # Measurement Noise Covariance (R) -> Sensor noise std=5.0
    R = np.array([[5.0**2]])
    
    # Initialize the Filter
    kf = KalmanFilter(mu0, Sigma0)
    
    # The "True System" scenario
    true_pos = 20.0
    true_vel = 15.0
    
    print("\nStarting Simulation (True Velocity = 15.0 m/s)...")
    print(f"{'Step':<5} | {'True Pos':<10} | {'Measured':<10} | {'Est Pos':<10} | {'Est Vel':<10}")
    print("-" * 55)
    
    for k in range(1, 11):
        # 1. True System moves (with some actual environmental noise)
        true_pos = true_pos + true_vel + np.random.normal(0, 2.0)
        true_vel = true_vel + np.random.normal(0, 0.5)
        
        # 2. Sensor takes noisy measurement
        measurement = np.array([true_pos + np.random.normal(0, 5.0)])
        
        # 3. Kalman Filter PREDICT
        kf.predict(A=A, Q=Q)
        
        # 4. Kalman Filter UPDATE
        mu_post, sigma_post = kf.update(y_md=measurement, C=C, R=R)
        
        print(f"{k:<5} | {true_pos:<10.2f} | {measurement[0]:<10.2f} | {mu_post[0]:<10.2f} | {mu_post[1]:<10.2f}")
    
    print("\nNotice how the estimated velocity rapidly converges to ~15.0 m/s,")
    print("even though the sensor ONLY measures position!")