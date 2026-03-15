"""
# ABOUTME: Tests for particle filter implementations  
# ABOUTME: Validates systematic resampling and bootstrap particle filter

Test particle filter implementations.
"""

import numpy as np
import pytest
from digital_twins.assimilation.particle import systematic_resampling, BootstrapParticleFilter


def test_systematic_resampling_returns_correct_count():
    """Test that systematic resampling returns exactly N indices."""
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    weights = weights / np.sum(weights)  # Normalize
    
    N = 100
    indices = systematic_resampling(weights, N)
    
    assert len(indices) == N
    assert np.all(indices >= 0)
    assert np.all(indices < len(weights))


def test_systematic_resampling_favors_higher_weights():
    """Test that particles with higher weights are selected more often."""
    weights = np.array([0.1, 0.1, 0.1, 0.7])
    weights = weights / np.sum(weights)
    
    N = 1000
    indices = systematic_resampling(weights, N)
    
    # Count how many times each particle was selected
    counts = np.bincount(indices, minlength=len(weights))
    
    # Particle 3 (weight=0.7) should be selected approximately 700 times
    assert counts[3] > 600
    assert counts[3] < 800
    
    # Other particles should be selected less
    for i in range(3):
        assert counts[i] < 200


def test_systematic_resampling_handles_uniform_weights():
    """Test resampling with uniform weights."""
    weights = np.ones(4) / 4
    
    N = 100
    indices = systematic_resampling(weights, N)
    
    counts = np.bincount(indices, minlength=len(weights))
    
    # Each particle should be selected approximately equally
    for count in counts:
        assert count > 15  # At least 15 times
        assert count < 35  # At most 35 times


def test_systematic_resampling_handles_single_particle():
    """Test resampling with single dominant particle."""
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    
    N = 100
    indices = systematic_resampling(weights, N)
    
    # All indices should be 0
    assert np.all(indices == 0)


def test_bootstrap_particle_filter_initialization():
    """Test BootstrapParticleFilter initialization."""
    N_particles = 100
    initial_particles = np.random.randn(N_particles, 2)
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    assert pf.N == N_particles
    assert pf.state_dim == 2
    assert np.allclose(pf.particles, initial_particles)
    assert len(pf.weights) == N_particles
    assert np.allclose(pf.weights, np.ones(N_particles) / N_particles)


def test_bootstrap_particle_filter_predict():
    """Test BootstrapParticleFilter prediction step."""
    N_particles = 50
    initial_particles = np.zeros((N_particles, 2))
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    # Simple transition model
    def f_func(x, u):
        return x + np.array([1.0, 0.5])
    
    process_noise_std = np.array([0.1, 0.1])
    
    particles = pf.predict(f_func, process_noise_std)
    
    assert particles.shape == (N_particles, 2)
    
    # Mean should be approximately [1.0, 0.5]
    mean_state = np.mean(particles, axis=0)
    assert np.abs(mean_state[0] - 1.0) < 0.2
    assert np.abs(mean_state[1] - 0.5) < 0.2
    
    # Should have some spread due to noise
    std_state = np.std(particles, axis=0)
    assert std_state[0] > 0.05
    assert std_state[1] > 0.05


def test_bootstrap_particle_filter_update():
    """Test BootstrapParticleFilter update step."""
    N_particles = 100
    # Create particles spread around different positions
    initial_particles = np.column_stack([
        np.linspace(-5, 5, N_particles),
        np.zeros(N_particles)
    ])
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    # Measurement function
    def g_func(x):
        return x[0]  # Observe position directly
    
    sensor_noise_std = 0.5
    y_md = 2.0  # Measurement at position 2
    
    particles = pf.update(y_md, g_func, sensor_noise_std)
    
    assert particles.shape == (N_particles, 2)
    
    # Particles should concentrate around measurement
    mean_pos = np.mean(particles[:, 0])
    assert np.abs(mean_pos - y_md) < 1.0
    
    # Variance should decrease after update
    std_pos = np.std(particles[:, 0])
    original_std = np.std(initial_particles[:, 0])
    assert std_pos < original_std


def test_bootstrap_particle_filter_rejuvenate():
    """Test particle rejuvenation."""
    N_particles = 50
    # All particles at same location (degenerate case)
    initial_particles = np.ones((N_particles, 2))
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    # Before rejuvenation, all particles are identical
    assert np.std(pf.particles) == 0
    
    # Apply rejuvenation
    rejuvenation_std = np.array([0.1, 0.1])
    pf.rejuvenate(rejuvenation_std)
    
    # After rejuvenation, particles should have spread
    assert np.std(pf.particles[:, 0]) > 0.05
    assert np.std(pf.particles[:, 1]) > 0.05
    
    # Mean should still be approximately the same
    mean_state = np.mean(pf.particles, axis=0)
    assert np.abs(mean_state[0] - 1.0) < 0.1
    assert np.abs(mean_state[1] - 1.0) < 0.1


def test_bootstrap_particle_filter_estimate_state():
    """Test state estimation from particles."""
    N_particles = 100
    particles = np.random.randn(N_particles, 2)
    particles[:, 0] = particles[:, 0] * 2 + 5  # Position centered at 5
    particles[:, 1] = particles[:, 1] * 0.5 + 2  # Velocity centered at 2
    
    pf = BootstrapParticleFilter(N_particles, particles)
    
    mean_state, std_state = pf.estimate_state()
    
    assert mean_state.shape == (2,)
    assert std_state.shape == (2,)
    
    # Check mean estimates
    assert np.abs(mean_state[0] - 5.0) < 0.5
    assert np.abs(mean_state[1] - 2.0) < 0.2
    
    # Check std estimates
    assert np.abs(std_state[0] - 2.0) < 0.5
    assert np.abs(std_state[1] - 0.5) < 0.2


def test_bootstrap_particle_filter_handles_zero_weights():
    """Test that filter handles case where all weights become zero."""
    N_particles = 50
    initial_particles = np.zeros((N_particles, 1))
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    # Measurement function that will make all likelihoods very small
    def g_func(x):
        return x[0]
    
    sensor_noise_std = 0.0001  # Very small noise
    y_md = 1000.0  # Measurement far from all particles
    
    # This should not crash, weights should be reset to uniform
    particles = pf.update(y_md, g_func, sensor_noise_std)
    
    assert particles.shape == (N_particles, 1)
    # Weights should be uniform after reset
    assert np.allclose(pf.weights, np.ones(N_particles) / N_particles)


def test_bootstrap_particle_filter_convergence():
    """Test that particle filter converges to true state over time."""
    N_particles = 200
    
    # True system state
    true_pos = 0.0
    true_vel = 1.0
    
    # Initialize particles with uncertainty
    initial_particles = np.column_stack([
        np.random.uniform(-10, 10, N_particles),
        np.random.uniform(-2, 2, N_particles)
    ])
    
    pf = BootstrapParticleFilter(N_particles, initial_particles)
    
    # Transition model
    dt = 0.1
    def f_func(x, u):
        return np.array([x[0] + x[1] * dt, x[1]])
    
    # Measurement model
    def g_func(x):
        return x[0]
    
    process_noise_std = np.array([0.01, 0.01])
    sensor_noise_std = 0.1
    
    # Run multiple steps
    for step in range(20):
        # True system evolves
        true_pos += true_vel * dt
        
        # Predict
        pf.predict(f_func, process_noise_std)
        
        # Generate noisy measurement
        measurement = true_pos + np.random.normal(0, sensor_noise_std)
        
        # Update
        pf.update(measurement, g_func, sensor_noise_std)
    
    # After convergence, estimate should be close to true state
    mean_state, _ = pf.estimate_state()
    assert np.abs(mean_state[0] - true_pos) < 0.6  # Slightly more tolerant
    assert np.abs(mean_state[1] - true_vel) < 0.3  # Slightly more tolerant


def test_systematic_resampling_deterministic():
    """Test that systematic resampling is more deterministic than random."""
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    weights = weights / np.sum(weights)
    
    N = 100
    
    # Run resampling multiple times
    results = []
    for _ in range(10):
        indices = systematic_resampling(weights, N)
        counts = np.bincount(indices, minlength=len(weights))
        results.append(counts)
    
    results = np.array(results)
    
    # Systematic resampling should have low variance across runs
    variances = np.var(results, axis=0)
    
    # Variance should be relatively low for systematic resampling
    assert np.all(variances < 20)