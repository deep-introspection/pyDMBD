"""
pytest configuration file for pyDMBD test suite.
"""

import numpy as np
import pytest
import torch


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_samples = 100
    n_features = 10
    X = torch.randn(n_samples, n_features)
    X = X - X.mean(0, True)
    return X


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 500
    n_features = 10
    n_targets = 4
    
    X = torch.randn(n_samples, n_features)
    X = X - X.mean(0, True)
    W = 2 * torch.randn(n_features, n_targets) / np.sqrt(10)
    Y = X @ W + torch.randn(n_samples, n_targets) / 100.0
    Y = Y + 0.5
    
    return X, Y, W


@pytest.fixture
def flocking_data():
    """Load or create synthetic flocking data for testing."""
    try:
        # Try to load real flocking data
        import numpy as np
        with np.load("data/couzin2zone_sim_hist_key1_100runs.npz") as data:
            r = data["r"]
            v = r[:, 1:] - r[:, :-1]  # velocity as position difference
            r = r[:, :-1]  # trim position to match velocity
            
        r = torch.tensor(r).float().swapaxes(0, 1)
        v = torch.tensor(v).float().swapaxes(0, 1)
        data = torch.cat((r, v), dim=-1)
        
        return data
    except FileNotFoundError:
        # Return synthetic flocking data that mimics the real structure
        n_timesteps = 100
        n_runs = 20  # number of simulation runs
        n_agents = 20  # number of flocking agents
        n_features = 4  # 2D position + 2D velocity
        
        torch.manual_seed(42)
        # Generate correlated position and velocity data
        data = torch.randn(n_timesteps, n_runs, n_agents, n_features)
        
        # Make positions and velocities somewhat correlated
        data[..., 2:] = 0.3 * data[..., :2] + 0.7 * data[..., 2:]
        
        # Apply some normalization
        data = data / data.std()
        
        return data


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require external data files"
    )
