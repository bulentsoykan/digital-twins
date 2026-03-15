"""
# ABOUTME: Tests for Kalman filter implementations
# ABOUTME: Validates KF, EKF, and EnKF algorithms

Test Kalman filter implementations.
"""

import numpy as np
import pytest
from digital_twins.assimilation.kalman import KalmanFilter, ExtendedKalmanFilter, EnsembleKalmanFilter


def test_kalman_filter_initialization():
    """Test KalmanFilter initialization."""
    mu0 = np.array([0.0, 1.0])
    Sigma0 = np.eye(2)
    
    kf = KalmanFilter(mu0, Sigma0)
    
    assert np.allclose(kf.mu, mu0)
    assert np.allclose(kf.Sigma, Sigma0)
    assert kf.n == 2


def test_kalman_filter_predict_step():
    """Test KalmanFilter prediction step."""
    mu0 = np.array([0.0, 0.0])
    Sigma0 = np.eye(2)
    
    kf = KalmanFilter(mu0, Sigma0)
    
    # Simple state transition: x_new = A @ x_old
    A = np.array([[1.0, 0.1], [0.0, 1.0]])  # Position-velocity model
    Q = 0.01 * np.eye(2)  # Process noise
    
    mu_pred, Sigma_pred = kf.predict(A, Q)
    
    # Check dimensions
    assert mu_pred.shape == (2,)
    assert Sigma_pred.shape == (2, 2)
    
    # After prediction, mean should be A @ mu0 = [0, 0]
    assert np.allclose(mu_pred, np.array([0.0, 0.0]))
    
    # Covariance should increase due to process noise
    assert np.all(np.diag(Sigma_pred) >= np.diag(Sigma0))


def test_kalman_filter_update_step():
    """Test KalmanFilter update step."""
    mu0 = np.array([10.0, 0.0])
    Sigma0 = np.array([[100.0, 0.0], [0.0, 100.0]])
    
    kf = KalmanFilter(mu0, Sigma0)
    
    # Measurement model: we observe position only
    C = np.array([[1.0, 0.0]])
    R = np.array([[1.0]])  # Measurement noise
    
    # Simulated measurement
    y_md = np.array([15.0])
    
    mu_post, Sigma_post = kf.update(y_md, C, R)
    
    # Check dimensions
    assert mu_post.shape == (2,)
    assert Sigma_post.shape == (2, 2)
    
    # Posterior mean should move toward measurement
    assert mu_post[0] > mu0[0]  # Should move from 10 toward 15
    assert mu_post[0] < y_md[0]  # But not all the way
    
    # Posterior covariance should decrease (information gain)
    assert Sigma_post[0, 0] < Sigma0[0, 0]


def test_kalman_filter_with_control_input():
    """Test KalmanFilter with control input."""
    mu0 = np.array([0.0, 0.0])
    Sigma0 = np.eye(2)
    
    kf = KalmanFilter(mu0, Sigma0)
    
    A = np.eye(2)
    B = np.array([[0.0], [1.0]])  # Control affects velocity
    u = np.array([2.0])  # Control input
    Q = 0.01 * np.eye(2)
    
    mu_pred, _ = kf.predict(A, Q, B, u)
    
    # With control input, velocity should increase
    assert np.allclose(mu_pred, np.array([0.0, 2.0]))


def test_kalman_filter_convergence():
    """Test that KalmanFilter converges over multiple steps."""
    # True system state
    true_state = np.array([100.0, 5.0])
    
    # Initialize with high uncertainty
    mu0 = np.array([0.0, 0.0])
    Sigma0 = 1000.0 * np.eye(2)
    
    kf = KalmanFilter(mu0, Sigma0)
    
    # System matrices
    dt = 1.0
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])  # Observe position only
    Q = 0.1 * np.eye(2)
    R = np.array([[10.0]])
    
    # Run multiple update cycles
    for _ in range(50):
        # Predict
        kf.predict(A, Q)
        
        # Generate noisy measurement from true state
        true_pos = true_state[0] + true_state[1] * dt
        true_state[0] = true_pos
        measurement = np.array([true_pos + np.random.normal(0, np.sqrt(R[0, 0]))])
        
        # Update
        kf.update(measurement, C, R)
    
    # After many updates, should converge close to true velocity
    assert np.abs(kf.mu[1] - 5.0) < 1.0


def test_extended_kalman_filter():
    """Test ExtendedKalmanFilter with nonlinear functions."""
    mu0 = np.array([1.0, 1.0])
    Sigma0 = np.eye(2)
    
    ekf = ExtendedKalmanFilter(mu0, Sigma0)
    
    # Nonlinear state transition
    def f_func(x, u):
        return np.array([x[0] + 0.1 * x[1], x[1]])
    
    # Jacobian of f
    F_jacobian = np.array([[1.0, 0.1], [0.0, 1.0]])
    Q = 0.01 * np.eye(2)
    
    mu_pred, Sigma_pred = ekf.predict(f_func, F_jacobian, Q)
    
    assert mu_pred.shape == (2,)
    assert Sigma_pred.shape == (2, 2)
    
    # Nonlinear measurement function
    def g_func(x):
        return np.array([x[0]**2])  # Observe squared position
    
    # Jacobian of g at current state
    G_jacobian = np.array([[2.0 * ekf.mu[0], 0.0]])
    R = np.array([[0.1]])
    y_md = np.array([1.5])
    
    mu_post, Sigma_post = ekf.update(y_md, g_func, G_jacobian, R)
    
    assert mu_post.shape == (2,)
    assert Sigma_post.shape == (2, 2)


def test_ensemble_kalman_filter_initialization():
    """Test EnsembleKalmanFilter initialization."""
    N_particles = 100
    mu0 = np.array([0.0, 1.0])
    Sigma0 = np.eye(2)
    
    enkf = EnsembleKalmanFilter(N_particles, mu0, Sigma0)
    
    assert enkf.N == N_particles
    assert enkf.n == 2
    assert enkf.ensemble.shape == (N_particles, 2)
    
    # Ensemble mean should be close to mu0
    ensemble_mean = np.mean(enkf.ensemble, axis=0)
    assert np.allclose(ensemble_mean, mu0, atol=0.5)


def test_ensemble_kalman_filter_predict():
    """Test EnsembleKalmanFilter prediction."""
    N_particles = 100
    mu0 = np.array([0.0, 0.0])
    Sigma0 = 0.01 * np.eye(2)
    
    enkf = EnsembleKalmanFilter(N_particles, mu0, Sigma0)
    
    # Nonlinear state transition
    def f_func(x, u):
        return np.array([x[0] + 0.1, x[1] + 0.05])
    
    Q = 0.001 * np.eye(2)
    
    ensemble = enkf.predict(f_func, Q)
    
    assert ensemble.shape == (N_particles, 2)
    
    # Mean should have moved according to f_func
    new_mean = np.mean(ensemble, axis=0)
    assert np.abs(new_mean[0] - 0.1) < 0.05
    assert np.abs(new_mean[1] - 0.05) < 0.05


def test_ensemble_kalman_filter_update():
    """Test EnsembleKalmanFilter update step."""
    N_particles = 200
    mu0 = np.array([0.0])
    Sigma0 = np.array([[1.0]])
    
    enkf = EnsembleKalmanFilter(N_particles, mu0, Sigma0)
    
    # Simple measurement function
    def g_func(x):
        return x[0]
    
    R = np.array([[0.1]])
    y_md = np.array([2.0])
    
    ensemble = enkf.update(y_md, g_func, R)
    
    assert ensemble.shape == (N_particles, 1)
    
    # Ensemble mean should move toward measurement
    new_mean = np.mean(ensemble)
    assert new_mean > mu0[0]
    assert new_mean < y_md[0]


def test_matrix_dimensions_consistency():
    """Test that matrix operations maintain proper dimensions."""
    # 3D state
    mu0 = np.array([1.0, 2.0, 3.0])
    Sigma0 = np.eye(3)
    
    kf = KalmanFilter(mu0, Sigma0)
    
    # State transition for 3D
    A = np.random.randn(3, 3)
    Q = 0.1 * np.eye(3)
    
    mu_pred, Sigma_pred = kf.predict(A, Q)
    assert mu_pred.shape == (3,)
    assert Sigma_pred.shape == (3, 3)
    
    # 2D measurement from 3D state
    C = np.random.randn(2, 3)
    R = 0.1 * np.eye(2)
    y_md = np.random.randn(2)
    
    mu_post, Sigma_post = kf.update(y_md, C, R)
    assert mu_post.shape == (3,)
    assert Sigma_post.shape == (3, 3)