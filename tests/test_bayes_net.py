"""
Bayesian Network Test Suite.

This file tests the BayesNet model and related distributions including
MatrixNormalWishart for regression and prediction tasks using pytest framework.
"""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from models.BayesNet import BayesNet
from models.dists import MatrixNormalWishart, MultivariateNormal_vector_format


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data for testing."""
    n = 4
    p = 10
    num_samples = 500
    
    X = torch.randn(num_samples, p)
    X = X - X.mean(0, True)
    W = 2 * torch.randn(p, n) / np.sqrt(10)
    Y = X @ W + torch.randn(num_samples, n) / 100.0
    Y = Y + 0.5
    
    return X, Y, W, n, p


@pytest.fixture
def model_configs():
    """Model configuration parameters."""
    return {
        'hidden_dims': (10, 10, 10),
        'latent_dims': (2, 2, 2),
        'iters': 100
    }


def test_matrix_normal_wishart_basic(synthetic_regression_data, random_seed):
    """Test basic MatrixNormalWishart functionality."""
    X, Y, W_true, n, p = synthetic_regression_data
    
    # Test MatrixNormalWishart
    W_true = W_true.transpose(-2, -1)
    W_hat = MatrixNormalWishart(mu_0=torch.zeros(n, p), pad_X=True)
    
    # Perform updates
    W_hat.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
    W_hat.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
    
    # Test prediction
    Y_hat = W_hat.predict(X.unsqueeze(-1))[0]
    MSE = ((Y - Y_hat.squeeze(-1)) ** 2).mean()
    
    # Basic assertions
    assert W_hat.mean() is not None, "W_hat should have valid mean"
    assert Y_hat.shape == Y.unsqueeze(-1).shape, "Prediction shape should match target"
    assert torch.isfinite(MSE), "MSE should be finite"
    assert MSE < 10.0, "MSE should be reasonable for synthetic data"


def test_matrix_normal_wishart_backward(synthetic_regression_data, random_seed):
    """Test MatrixNormalWishart backward prediction."""
    X, Y, W_true, n, p = synthetic_regression_data
    
    W_true = W_true.transpose(-2, -1)
    W_hat = MatrixNormalWishart(mu_0=torch.zeros(n, p), pad_X=True)
    W_hat.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
    
    # Test backward prediction
    pY = MultivariateNormal_vector_format(
        mu=Y.unsqueeze(-1), invSigma=1000 * torch.eye(n)
    )
    px, Res = W_hat.backward(pY)
    
    # Test Elog_like_X
    invSigma_x_x, invSigmamu_x, Residual = W_hat.Elog_like_X(Y.unsqueeze(-1))
    mu_x = (invSigma_x_x.inverse() @ invSigmamu_x)
    
    # Assertions
    assert px is not None, "Backward prediction should return valid distribution"
    assert invSigma_x_x.shape[-1] == p, "Precision matrix should have correct dimensions"
    assert mu_x.shape[-2] == p, "Predicted mean should have correct dimensions"
    assert torch.isfinite(mu_x).all(), "Backward prediction should be finite"


def test_matrix_normal_wishart_forward_prediction(synthetic_regression_data, random_seed):
    """Test MatrixNormalWishart forward prediction methods."""
    X, Y, W_true, n, p = synthetic_regression_data
    
    W_true = W_true.transpose(-2, -1)
    W_hat = MatrixNormalWishart(mu_0=torch.zeros(n, p), pad_X=True)
    W_hat.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
    
    # Test forward prediction
    Y_hat2 = W_hat.forward(
        MultivariateNormal_vector_format(
            mu=X.unsqueeze(-1), Sigma=torch.eye(p) / 1000.0
        )
    ).mean().squeeze(-1)
    
    # Compare with direct prediction
    Y_hat1 = W_hat.predict(X.unsqueeze(-1))[0].squeeze(-1)
    
    assert Y_hat1.shape == Y.shape, "Direct prediction should match target shape"
    assert Y_hat2.shape == Y.shape, "Forward prediction should match target shape"
    assert torch.isfinite(Y_hat1).all(), "Direct prediction should be finite"
    assert torch.isfinite(Y_hat2).all(), "Forward prediction should be finite"
    
    # The two methods should give similar results
    correlation = torch.corrcoef(torch.stack([Y_hat1.flatten(), Y_hat2.flatten()]))[0, 1]
    assert correlation > 0.9, "Different prediction methods should be highly correlated"


@pytest.mark.slow
def test_bayes_net_training(synthetic_regression_data, model_configs, random_seed):
    """Test BayesNet model training."""
    X, Y, W_true, n, p = synthetic_regression_data
    hidden_dims = model_configs['hidden_dims']
    latent_dims = model_configs['latent_dims']
    iters = model_configs['iters']
    
    # Test standard BayesNet
    model = BayesNet(n, p, hidden_dims, latent_dims)
    model.update(X, Y, lr=1, iters=iters, verbose=False, FBI=False)
    
    # Test predictions
    Yhat = model.predict(X)
    final_mse = model.MSE[-1]
    
    # Basic assertions
    assert len(model.ELBO_save) > 0, "ELBO should be tracked during training"
    assert len(model.MSE) > 0, "MSE should be tracked during training"
    assert Yhat.shape == Y.unsqueeze(-1).shape, "Prediction shape should match target"
    assert final_mse < np.inf, "Final MSE should be finite"
    assert final_mse > 0, "Final MSE should be positive"


@pytest.mark.slow
def test_bayes_net_fbi_comparison(synthetic_regression_data, model_configs, random_seed):
    """Test BayesNet with and without FBI (Full Bayesian Inference)."""
    X, Y, W_true, n, p = synthetic_regression_data
    hidden_dims = model_configs['hidden_dims']
    latent_dims = model_configs['latent_dims']
    iters = model_configs['iters']
    
    # Test standard BayesNet
    model = BayesNet(n, p, hidden_dims, latent_dims)
    model.update(X, Y, lr=1, iters=iters, verbose=False, FBI=False)
    
    # Test BayesNet with FBI
    set_model = BayesNet(n, p, hidden_dims, latent_dims)
    set_model.update(X, Y, lr=1, iters=iters, verbose=False, FBI=True)
    
    # Get predictions
    Yhat = model.predict(X)
    set_Yhat = set_model.predict(X)
    
    # Compute effective weights
    W_net = torch.eye(X.shape[-1])
    for k, layer in enumerate(model.layers):
        W_net = layer.weights() @ W_net
        
    set_W_net = torch.eye(X.shape[-1])
    for k, layer in enumerate(set_model.layers):
        set_W_net = layer.weights() @ set_W_net
    
    # Assertions
    assert model.MSE[-1] > 0, "Standard model should have positive MSE"
    assert set_model.MSE[-1] > 0, "FBI model should have positive MSE"
    assert torch.isfinite(W_net).all(), "Standard model weights should be finite"
    assert torch.isfinite(set_W_net).all(), "FBI model weights should be finite"
    assert Yhat.shape == set_Yhat.shape, "Both models should produce same prediction shape"
    
    # Both models should produce reasonable predictions
    mse_standard = ((Y - Yhat.squeeze(-1)) ** 2).mean()
    mse_fbi = ((Y - set_Yhat.squeeze(-1)) ** 2).mean()
    assert mse_standard < 10.0, "Standard model should have reasonable MSE"
    assert mse_fbi < 10.0, "FBI model should have reasonable MSE"


def test_bayes_net_weights_extraction(synthetic_regression_data, model_configs, random_seed):
    """Test that we can extract meaningful weights from BayesNet."""
    X, Y, W_true, n, p = synthetic_regression_data
    hidden_dims = (5, 5)  # Smaller for faster testing
    latent_dims = (2, 2)
    
    model = BayesNet(n, p, hidden_dims, latent_dims)
    model.update(X, Y, lr=1, iters=20, verbose=False, FBI=False)  # Fewer iterations
    
    # Extract layer weights
    assert len(model.layers) > 0, "Model should have layers"
    
    for i, layer in enumerate(model.layers):
        weights = layer.weights()
        assert weights is not None, f"Layer {i} should have weights"
        assert torch.isfinite(weights).all(), f"Layer {i} weights should be finite"
        assert weights.shape[0] > 0, f"Layer {i} weights should have valid shape"


def test_bayes_net_elbo_convergence(synthetic_regression_data, random_seed):
    """Test that ELBO generally improves during training."""
    X, Y, W_true, n, p = synthetic_regression_data
    hidden_dims = (5, 5)  # Smaller for faster testing
    latent_dims = (2, 2)
    
    model = BayesNet(n, p, hidden_dims, latent_dims)
    model.update(X, Y, lr=1, iters=30, verbose=False, FBI=False)
    
    # Check ELBO trajectory
    elbo_values = torch.tensor(model.ELBO_save)
    
    assert len(elbo_values) > 10, "Should have multiple ELBO values"
    assert torch.isfinite(elbo_values).all(), "All ELBO values should be finite"
    
    # ELBO should generally increase (allow for some fluctuation)
    elbo_end = elbo_values[-5:].mean()
    elbo_start = elbo_values[:5].mean()
    assert elbo_end > elbo_start - 1.0, "ELBO should generally improve during training"


def test_bayes_net_prediction_consistency(synthetic_regression_data, random_seed):
    """Test that predictions are consistent across multiple calls."""
    X, Y, W_true, n, p = synthetic_regression_data
    hidden_dims = (5, 5)
    latent_dims = (2, 2)
    
    model = BayesNet(n, p, hidden_dims, latent_dims)
    model.update(X, Y, lr=1, iters=10, verbose=False, FBI=False)
    
    # Get multiple predictions
    pred1 = model.predict(X[:10])
    pred2 = model.predict(X[:10])
    
    # Predictions should be identical (deterministic given same input)
    assert torch.allclose(pred1, pred2, atol=1e-6), "Predictions should be consistent"


def test_matrix_normal_wishart_initialization():
    """Test MatrixNormalWishart initialization with different parameters."""
    n, p = 3, 5
    
    # Test basic initialization
    mnw1 = MatrixNormalWishart(mu_0=torch.zeros(n, p), pad_X=True)
    assert mnw1 is not None, "Basic initialization should work"
    
    # Test with custom precision matrices
    mnw2 = MatrixNormalWishart(
        mu_0=torch.zeros(n, p),
        Omega_0=torch.eye(n),
        Lambda_0=torch.eye(p),
        pad_X=False
    )
    assert mnw2 is not None, "Custom precision initialization should work"
    
    # Test mean extraction
    mean1 = mnw1.mean()
    mean2 = mnw2.mean()
    
    assert mean1.shape[-2:] == (n, p) or mean1.shape[-2:] == (n, p+1), "Mean should have correct shape"
    assert mean2.shape[-2:] == (n, p), "Mean should have correct shape"
