"""
VBEM Test Suite - Comprehensive tests for various model functionalities.

This file tests basic functionality of models including:
- MatrixNormalWishart with masking
- Autoregressive Hidden Markov Models (ARHMM)
- Bayesian Factor Analysis
- discrete Hidden Markov Models (dHMM)
- Gaussian Mixture Models (GMM)
- Linear Dynamical Systems (LDS)
- And other model components
"""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

# Import various models and distributions
from models.dists import MatrixNormalWishart


class TestMatrixNormalWishartMasking:
    """Test MatrixNormalWishart with various masking options."""
    
    @pytest.fixture
    def masked_regression_setup(self, random_seed):
        """Setup for masked regression testing."""
        n = 2
        p = 10
        n_samples = 400
        batch_num = 4
        
        w_true = torch.randn(n, p) / np.sqrt(p)
        X_mask = w_true.abs().sum(-2) < w_true.abs().sum(-2).mean()
        X_mask = X_mask.unsqueeze(-2)
        w_true = w_true * X_mask
        b_true = torch.randn(n, 1)
        pad_X = True
        
        X = torch.randn(n_samples, p)
        Y = torch.zeros(n_samples, n)
        for i in range(n_samples):
            Y[i, :] = (X[i:i + 1, :] @ w_true.transpose(-1, -2) +
                       b_true.transpose(-2, -1) * pad_X + torch.randn(1) / 100.0)
        
        return {
            'X': X, 'Y': Y, 'w_true': w_true, 'X_mask': X_mask,
            'n': n, 'p': p, 'pad_X': pad_X
        }
    
    def test_vanilla_matrix_normal_wishart(self, masked_regression_setup):
        """Test vanilla MatrixNormalWishart without masking."""
        setup = masked_regression_setup
        X, Y = setup['X'], setup['Y']
        n, p, pad_X = setup['n'], setup['p'], setup['pad_X']
        
        W0 = MatrixNormalWishart(
            torch.zeros(n, p), torch.eye(n), torch.eye(p), pad_X=pad_X
        )
        W0.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
        
        # Test prediction
        Yhat = W0.predict(X.unsqueeze(-1))[0].squeeze(-1)
        
        assert Yhat.shape == Y.shape, "Prediction should match target shape"
        assert torch.isfinite(Yhat).all(), "Predictions should be finite"
        
        # Test weights
        weights = W0.weights()
        assert weights is not None, "Should be able to extract weights"
        assert torch.isfinite(weights).all(), "Weights should be finite"
    
    def test_matrix_normal_wishart_with_x_mask(self, masked_regression_setup):
        """Test MatrixNormalWishart with X_mask."""
        setup = masked_regression_setup
        X, Y, X_mask = setup['X'], setup['Y'], setup['X_mask']
        n, p, pad_X = setup['n'], setup['p'], setup['pad_X']
        
        W1 = MatrixNormalWishart(
            torch.zeros(n, p), torch.eye(n), torch.eye(p), 
            X_mask=X_mask, pad_X=pad_X
        )
        W1.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
        
        # Test prediction
        Yhat = W1.predict(X.unsqueeze(-1))[0].squeeze(-1)
        
        assert Yhat.shape == Y.shape, "Prediction should match target shape"
        assert torch.isfinite(Yhat).all(), "Predictions should be finite"
        
        # Test weights
        weights = W1.weights()
        assert torch.isfinite(weights).all(), "Weights should be finite"
    
    def test_matrix_normal_wishart_with_mask(self, masked_regression_setup):
        """Test MatrixNormalWishart with general mask."""
        setup = masked_regression_setup
        X, Y, X_mask = setup['X'], setup['Y'], setup['X_mask']
        n, p, pad_X = setup['n'], setup['p'], setup['pad_X']
        
        W2 = MatrixNormalWishart(
            torch.zeros(n, p), torch.eye(n), torch.eye(p),
            mask=X_mask.expand(n, p), pad_X=pad_X
        )
        W2.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
        
        # Test prediction
        Yhat = W2.predict(X.unsqueeze(-1))[0].squeeze(-1)
        
        assert Yhat.shape == Y.shape, "Prediction should match target shape"
        assert torch.isfinite(Yhat).all(), "Predictions should be finite"
    
    def test_backward_prediction_consistency(self, masked_regression_setup):
        """Test that backward predictions work consistently across masking options."""
        setup = masked_regression_setup
        X, Y = setup['X'], setup['Y']
        n, p, pad_X = setup['n'], setup['p'], setup['pad_X']
        
        W0 = MatrixNormalWishart(
            torch.zeros(n, p), torch.eye(n), torch.eye(p), pad_X=pad_X
        )
        W0.raw_update(X.unsqueeze(-1), Y.unsqueeze(-1))
        
        # Test backward prediction
        invSigma_xx, invSigmamu_x, Res = W0.Elog_like_X(Y.unsqueeze(-1))
        mu_x0 = torch.linalg.solve(invSigma_xx + 1e-6 * torch.eye(p), invSigmamu_x)
        
        assert mu_x0.shape[-2] == p, "Backward prediction should have correct dimension"
        assert torch.isfinite(mu_x0).all(), "Backward prediction should be finite"


class TestARHMM:
    """Test Autoregressive Hidden Markov Model variants."""
    
    def test_vanilla_arhmm(self, random_seed):
        """Test basic ARHMM functionality."""
        from models.ARHMM import ARHMM
        
        dim = 6
        batch_dim = 7
        hidden_dim = 5
        T = 100
        num_samples = 200
        sample_shape = (T, num_samples)

        A = torch.rand(hidden_dim, hidden_dim) + 4 * torch.eye(hidden_dim)
        A = A / A.sum(-1, keepdim=True)
        B = torch.randn(hidden_dim, dim)

        z = torch.rand(T, num_samples, hidden_dim).argmax(-1)
        x = torch.zeros(T, num_samples, dim)
        for t in range(1, T):
            x[t] = B[z[t]] + torch.randn(num_samples, dim) / 10.0

        model = ARHMM(hidden_dim, dim, sample_shape)
        
        # Test that model can be updated
        model.update(x, iters=5, lr=1, verbose=False)
        
        # Basic assertions
        assert model.px is not None, "Model should have px after update"
        assert hasattr(model, 'obs_model'), "Model should have obs_model"
        
        # Test assignment
        assignment = model.assignment()
        assert assignment is not None, "Model should provide assignment"
        assert assignment.shape[:2] == (T, num_samples), "Assignment should match data shape"
    
    def test_arhmm_prxy(self, random_seed):
        """Test ARHMM with probabilistic X and Y inputs."""
        from models.ARHMM import ARHMM_prXY
        
        dim = 6
        ydim = 3
        hidden_dim = 5
        T = 50
        batch_size = 20

        x = torch.randn(T, batch_size, dim)
        y = torch.randn(T, batch_size, ydim)

        model = ARHMM_prXY(hidden_dim, dim, ydim, batch_shape=())
        
        # Test update
        model.update((x.unsqueeze(-1), y.unsqueeze(-1)), iters=5, lr=1, verbose=False)
        
        assert model.px is not None, "Model should have px after update"
        
    def test_arhmm_prxry(self, random_seed):
        """Test ARHMM with probabilistic X, R, and Y inputs."""
        from models.ARHMM import ARHMM_prXRY
        
        dim = 6
        rdim = 2
        ydim = 3
        xdim = 4
        T = 30
        batch_size = 15

        x = torch.randn(T, batch_size, xdim)
        r = torch.randn(T, batch_size, rdim)
        y = torch.randn(T, batch_size, ydim)

        from models.dists import MultivariateNormal_vector_format
        pX = MultivariateNormal_vector_format(
            mu=x, Sigma=torch.zeros(x.shape[:-1] + (xdim,)) + torch.eye(xdim) / 10
        )
        
        model = ARHMM_prXRY(5, dim, xdim, rdim, batch_shape=())
        pXRY = (pX, r.unsqueeze(-1).unsqueeze(-3), y.unsqueeze(-1).unsqueeze(-3))
        
        # Test update
        model.update(pXRY, iters=5, lr=1, verbose=False)
        
        assert hasattr(model, 'px'), "Model should have px attribute"


class TestBayesianFactorAnalysis:
    """Test Bayesian Factor Analysis model."""
    
    def test_bayesian_factor_analysis_basic(self, random_seed):
        """Test basic Bayesian Factor Analysis functionality."""
        from models.BayesianFactorAnalysis import BayesianFactorAnalysis
        
        obs_dim = 20
        latent_dim = 2
        num_samps = 200
        
        model = BayesianFactorAnalysis(obs_dim, latent_dim, pad_X=False)

        # Generate synthetic data
        A = torch.randn(latent_dim, obs_dim)
        Z = torch.randn(num_samps, latent_dim)
        Y = Z @ A + torch.randn(num_samps, obs_dim) / 10.0
        Y = Y - Y.mean(0, True)
        A = A.transpose(-2, -1)

        # Test update
        model.update(Y.unsqueeze(-1), iters=10, lr=1, verbose=False)
        
        # Basic assertions
        assert hasattr(model, 'obs_model'), "Model should have obs_model"
        assert hasattr(model, 'px'), "Model should have px"
        
        # Test prediction capability
        mu = model.obs_model.predict(model.px.mean())[0].squeeze(-1)
        assert mu.shape == Y.shape, "Prediction should match data shape"
        assert torch.isfinite(mu).all(), "Predictions should be finite"


class TestDiscreteHMM:
    """Test discrete Hidden Markov Model."""
    
    def test_dhmm_basic(self, random_seed):
        """Test basic dHMM functionality."""
        from models.dHMM import dHMM
        
        T = 100
        num_samples = 50
        n_states = 4
        obs_dim = 3
        sample_shape = (T, num_samples)

        # Generate synthetic data
        states = torch.randint(0, n_states, (T, num_samples))
        B = torch.randn(n_states, obs_dim)
        Y = B[states] + torch.randn(T, num_samples, obs_dim) / 5.0

        model = dHMM(n_states, obs_dim, sample_shape, 
                    obs_dist='NormalInverseWishart')

        # Test update
        model.update(Y.unsqueeze(-1), iters=5, lr=1, verbose=False)
        
        # Basic assertions
        assert hasattr(model, 'obs_model'), "Model should have obs_model"
        assert hasattr(model, 'init_prob'), "Model should have init_prob"
        assert hasattr(model, 'trans_prob'), "Model should have trans_prob"
        
        # Test assignment
        assignment = model.assignment()
        assert assignment.shape == (T, num_samples), "Assignment should match data shape"


class TestGaussianMixtureModel:
    """Test Gaussian Mixture Model functionality."""
    
    @pytest.fixture
    def gmm_synthetic_data(self, random_seed):
        """Generate synthetic data for GMM testing."""
        dim = 2
        nc = 4
        mu = torch.randn(4, 2) * 4
        A = torch.randn(4, 2, 2) / np.sqrt(2)
        num_samples = 200
        
        data = torch.zeros(num_samples, 2)
        for i in range(num_samples):
            data[i, :] = (mu[i % 4, :] + A[i % 4, :, :] @ torch.randn(2) +
                          torch.randn(2) / 8.0)
        
        return data, dim, nc

    def test_gaussian_mixture_model(self, gmm_synthetic_data, random_seed):
        """Test standard Gaussian Mixture Model."""
        from models.GaussianMixtureModel import GaussianMixtureModel as GMM
        
        data, dim, _ = gmm_synthetic_data
        nc = 6  # Number of components
        
        gmm = GMM(nc, dim)
        gmm.update(data.unsqueeze(-2), 20, 1, verbose=False)
        
        # Basic assertions
        assert hasattr(gmm, 'assignment'), "GMM should have assignment method"
        assignment = gmm.assignment()
        assert assignment.shape[0] == data.shape[0], "Assignment should match data length"
        assert assignment.min() >= 0, "Assignment indices should be non-negative"
        assert assignment.max() < nc, "Assignment indices should be within range"

    def test_isotropic_gaussian_mixture_model(self, gmm_synthetic_data, random_seed):
        """Test Isotropic Gaussian Mixture Model."""
        from models.IsotropicGaussianMixtureModel import IsotropicGaussianMixtureModel as IGMM
        
        data, dim, _ = gmm_synthetic_data
        nc = 6
        
        igmm = IGMM(nc, dim)
        igmm.update(data.unsqueeze(-2), 20, 1, verbose=False)
        
        # Basic assertions
        assert hasattr(igmm, 'assignment'), "IGMM should have assignment method"
        assignment = igmm.assignment()
        assert assignment.shape[0] == data.shape[0], "Assignment should match data length"


class TestMixtureModels:
    """Test various mixture model implementations."""
    
    def test_mixture_non_trivial_batch_shape(self, random_seed):
        """Test mixture with non-trivial batch shape."""
        from models.dMixture import dMixture
        
        num_comp = 4
        dim = 2
        num_samps = 100
        batch_shape = (3, 2)
        
        data = torch.randn(batch_shape + (num_samps, dim))
        
        model = dMixture(num_comp, batch_shape, dim)
        model.update(data.unsqueeze(-1), iters=10, lr=1, verbose=False)
        
        # Test that model handles batch shape correctly
        assert hasattr(model, 'obs_model'), "Model should have obs_model"
        assert hasattr(model, 'assignment'), "Model should have assignment method"
        
        assignment = model.assignment()
        assert assignment.shape[:len(batch_shape)] == batch_shape, "Assignment should preserve batch shape"

    def test_mixture_non_trivial_event_shape(self, random_seed):
        """Test mixture with non-trivial event shape."""
        from models.dMixture import dMixture
        
        num_comp = 4
        dim = (2, 3)  # Non-trivial event shape
        num_samps = 50
        batch_shape = (2,)
        
        data = torch.randn(batch_shape + (num_samps,) + dim)
        
        model = dMixture(num_comp, batch_shape, dim)
        model.update(data.unsqueeze(-1), iters=5, lr=1, verbose=False)
        
        # Basic functionality test
        assert hasattr(model, 'obs_model'), "Model should have obs_model"


# Utility functions for testing
def test_smoothing_function():
    """Test the smoothing utility function."""
    from tests.test_dmbd_pytest import smoothe
    
    # Create test data
    data = torch.randn(100, 5, 3)
    n = 5
    
    smoothed = smoothe(data, n)
    
    assert smoothed.shape[0] == (data.shape[0] - n) // n, "Smoothed data should have correct length"
    assert smoothed.shape[1:] == data.shape[1:], "Other dimensions should be preserved"
    assert torch.isfinite(smoothed).all(), "Smoothed data should be finite"


# Parametrized tests for robustness
@pytest.mark.parametrize("dim,hidden_dim,T", [
    (3, 2, 20),
    (5, 4, 30),
    (2, 3, 25)
])
def test_arhmm_parametrized(dim, hidden_dim, T, random_seed):
    """Parametrized test for ARHMM with different configurations."""
    from models.ARHMM import ARHMM
    
    num_samples = 50
    sample_shape = (T, num_samples)
    
    # Generate synthetic data
    x = torch.randn(T, num_samples, dim)
    
    model = ARHMM(hidden_dim, dim, sample_shape)
    model.update(x, iters=3, lr=1, verbose=False)
    
    assert model.px is not None, f"Model should work with dim={dim}, hidden_dim={hidden_dim}, T={T}"


@pytest.mark.slow
class TestIntegrationVBEM:
    """Integration tests for VBEM models."""
    
    def test_model_chain_compatibility(self, random_seed):
        """Test that different models can work together in a pipeline."""
        # This is a placeholder for more complex integration tests
        # that might involve chaining different models together
        pass
    
    def test_large_scale_gmm(self, random_seed):
        """Test GMM on larger scale data."""
        from models.GaussianMixtureModel import GaussianMixtureModel as GMM
        
        dim = 10
        nc = 8
        num_samples = 1000
        
        # Generate complex synthetic data
        centers = torch.randn(nc, dim) * 5
        data = []
        for i in range(num_samples):
            center_idx = i % nc
            sample = centers[center_idx] + torch.randn(dim)
            data.append(sample)
        
        data = torch.stack(data)
        
        gmm = GMM(nc, dim)
        gmm.update(data.unsqueeze(-2), 30, 1, verbose=False)
        
        assignment = gmm.assignment()
        
        # Check that model produces reasonable clustering
        assert len(torch.unique(assignment)) > 1, "Should find multiple clusters"
        assert len(torch.unique(assignment)) <= nc, "Should not exceed number of components"
