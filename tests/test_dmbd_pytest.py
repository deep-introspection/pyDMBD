"""
DMBD Test Suite - Flocking Behavior Example

This file demonstrates the Dynamic Markov Blanket Discovery algorithm
on flocking behavior data using pytest framework.
"""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from models.DynamicMarkovBlanketDiscovery import DMBD, animate_results


def smoothe(data, n):
    """Smooth data by averaging over n consecutive time steps."""
    temp = data[0:-n]
    for i in range(1, n):
        temp = temp + data[i:-(n-i)]
    return temp[::n] / n


@pytest.fixture
def processed_flocking_data(flocking_data):
    """Process flocking data for DMBD testing."""
    # Apply smoothing if we have enough timesteps
    if flocking_data.shape[0] >= 20:
        data = 2 * smoothe(flocking_data, 20)
        data = data[:80]
    else:
        # For synthetic data, just take first part
        data = flocking_data[:min(80, flocking_data.shape[0])]
    
    # Take a subset of runs for testing
    data = data[:, :10]  # First 10 runs only
    
    return data


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_data
def test_dmbd_flocking_behavior(processed_flocking_data, random_seed):
    """Test DMBD algorithm on flocking behavior data."""
    data = processed_flocking_data
    
    # Initialize DMBD model with correct dimensions for flocking data
    model = DMBD(
        obs_shape=data.shape[-2:],
        role_dims=(2, 2, 2),
        hidden_dims=(4, 2, 2),
        regression_dim=-1,
        control_dim=0,
        number_of_objects=data.shape[-2],  # Use actual number of agents
        unique_obs=False
    )

    # Training loop with proper batch size constraints
    iters = 40
    max_batch_idx = min(50, data.shape[1] - 1)
    for i in range(iters):
        model.update(
            data[:, torch.randint(0, max_batch_idx, (min(10, data.shape[1]),))],
            None,
            None,
            iters=2,
            latent_iters=4,
            lr=0.05,
            verbose=False  # Set to False for testing
        )

    # Final update with proper batch constraints
    batch_size = min(4, data.shape[1])
    model.update(
        data[:, 0:batch_size], None, None,
        iters=1, latent_iters=8, lr=0.01, verbose=False  # Use small non-zero lr
    )
    
    # Verify model training
    assert model.px is not None, "Model px should be initialized after training"
    assert model.obs_model is not None, "Model obs_model should be initialized"
    
    # Extract results
    sbz = model.px.mean()
    B = model.obs_model.obs_dist.mean()
    if model.regression_dim == 1:
        roles = B[..., :-1] @ sbz + B[..., -1:]
    else:
        roles = B @ sbz
    sbz = sbz.squeeze(-3).squeeze(-1)
    roles = roles.squeeze(-1)[..., 0:2]

    # Verify shapes and basic properties
    assert sbz.shape[0] > 0, "sbz should have valid time dimension"
    assert roles.shape[-1] == 2, "roles should have 2D output"
    assert torch.isfinite(sbz).all(), "sbz should contain finite values"
    assert torch.isfinite(roles).all(), "roles should contain finite values"


@pytest.mark.slow 
@pytest.mark.requires_data
def test_dmbd_visualization(processed_flocking_data, random_seed):
    """Test DMBD visualization capabilities."""
    data = processed_flocking_data
    
    # Quick model training for visualization test
    model = DMBD(
        obs_shape=data.shape[-2:],
        role_dims=(2, 2, 2),
        hidden_dims=(4, 2, 2),
        regression_dim=-1,
        control_dim=0,
        number_of_objects=data.shape[-2],  # Use actual number of agents
        unique_obs=False
    )

    # Minimal training for testing
    max_batch_idx = min(50, data.shape[1] - 1)
    for i in range(5):  # Reduced iterations for testing
        model.update(
            data[:, torch.randint(0, max_batch_idx, (min(5, data.shape[1]),))],
            None,
            None,
            iters=1,
            latent_iters=2,
            lr=0.05,
            verbose=False
        )

    batch_size = min(4, data.shape[1])
    model.update(
        data[:, 0:batch_size], None, None,
        iters=1, latent_iters=2, lr=0.01, verbose=False  # Use small non-zero lr
    )
    
    # Test that we can extract visualization data
    sbz = model.px.mean()
    B = model.obs_model.obs_dist.mean()
    if model.regression_dim == 1:
        roles = B[..., :-1] @ sbz + B[..., -1:]
    else:
        roles = B @ sbz
    sbz = sbz.squeeze(-3).squeeze(-1)
    roles = roles.squeeze(-1)[..., 0:2]

    batch_num = 0
    temp1 = data[:, batch_num, :, 0]
    temp2 = data[:, batch_num, :, 1]
    rtemp1 = roles[:, batch_num, :, 0]
    rtemp2 = roles[:, batch_num, :, 1]

    # Test assignment functionality
    assignment = model.assignment()
    obs_assignment = model.obs_model.assignment()
    
    assert assignment is not None, "Model should provide assignment"
    assert obs_assignment is not None, "Obs model should provide assignment"
    assert assignment.shape[1] > batch_num, "Assignment should cover batch dimension"
    
    # Verify we can create indices for visualization
    idx = (assignment[:, batch_num, :] == 0)
    assert idx.dtype == torch.bool, "Assignment indices should be boolean"
    
    # Test that visualization data has reasonable properties
    ev_dim = model.role_dims[0]
    ob_dim = np.sum(model.role_dims[1:])
    
    assert ev_dim > 0, "Environment dimension should be positive"
    assert ob_dim > 0, "Object dimension should be positive"
    assert model.number_of_objects > 0, "Number of objects should be positive"


@pytest.mark.slow
@pytest.mark.requires_data 
def test_dmbd_animation_setup(processed_flocking_data, random_seed):
    """Test that animation setup works without actually creating movies."""
    data = processed_flocking_data
    
    # Quick model setup
    model = DMBD(
        obs_shape=data.shape[-2:],
        role_dims=(2, 2, 2),
        hidden_dims=(4, 2, 2),
        regression_dim=-1,
        control_dim=0,
        number_of_objects=data.shape[-2],  # Use actual number of agents
        unique_obs=False
    )

    # Minimal training with small learning rate to avoid numerical issues
    batch_size = min(4, data.shape[1])
    model.update(
        data[:, 0:batch_size], None, None,
        iters=1, latent_iters=1, lr=0.01, verbose=False  # Use small non-zero lr
    )
    
    # Test that animate_results can be instantiated
    ar = animate_results('role', None, xlim=(-0.2, 0.6), ylim=(-0.5, 2), fps=10)
    assert ar is not None, "animate_results should be instantiable"
    
    # Test that the model and data are compatible with animation
    # (without actually making the movie which would be slow)
    assert hasattr(ar, 'make_movie'), "animate_results should have make_movie method"


def test_dmbd_basic_initialization():
    """Test basic DMBD model initialization without data."""
    model = DMBD(
        obs_shape=(10, 4),
        role_dims=(2, 2, 2),
        hidden_dims=(4, 2, 2),
        regression_dim=-1,
        control_dim=0,
        number_of_objects=5,
        unique_obs=False
    )
    
    assert model is not None, "Model should be initialized"
    assert model.role_dims == (2, 2, 2), "Role dims should match initialization"
    assert model.hidden_dims == (4, 2, 2), "Hidden dims should match initialization"
    assert model.number_of_objects == 5, "Number of objects should match initialization"
