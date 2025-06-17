
"""
Flame Dataset Example for Dynamic Markov Blanket Discovery.

This script demonstrates the use of DMBD on the Flame dataset,
performing dynamic discovery and visualization of results.
"""

import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from models.DynamicMarkovBlanketDiscovery import DMBD

print('Test on Flame data set')

# Load the flame dataset
data = torch.load('./data/flame.pt').clone().detach()

# Initialize DMBD model with specified parameters
model = DMBD(
    obs_shape=data.shape[-2:],
    role_dims=(3, 3, 3),
    hidden_dims=(4, 4, 4),
    batch_shape=(),
    regression_dim=-1,
    control_dim=0,
    number_of_objects=1
)

# Setup colormap for visualization
cmap = ListedColormap(['red', 'green', 'blue'])
vmin = 0  # Minimum value of the color scale
vmax = 2  # Maximum value of the color scale
norm = Normalize(vmin=vmin, vmax=vmax)

# Training loop
for i in range(10):
    # Update model with data
    model.update(data, None, None, iters=2, latent_iters=1, lr=0.5)

    # Get model predictions and calculate dimensions
    sbz = model.px.mean().squeeze()
    r1 = model.role_dims[0]
    r2 = r1 + model.role_dims[1]
    r3 = r2 + model.role_dims[2]
    h1 = model.hidden_dims[0]
    h2 = h1 + model.hidden_dims[1]
    h3 = h2 + model.hidden_dims[2]

    # Generate and save assignment visualization
    p = model.assignment_pr()
    a = 2 - model.assignment()
    plt.imshow(
        a[:, 0, :].transpose(-2, -1),
        cmap=cmap,
        norm=norm,
        origin='lower'
    )
    plt.xlabel('Time')
    plt.ylabel('Location')
    plt.savefig('flame_assignments.png')

    # Calculate PC scores for each component
    p = p.sum(-2)
    print('Show PC scores')    

    # Extract and center components
    s = sbz[:, :, 0:h1]
    s = s - s.mean(0).mean(0)
    b = sbz[:, :, h1:h2]
    b = b - b.mean(0).mean(0)
    z = sbz[:, :, h2:h3]
    z = z - z.mean(0).mean(0)

    # Calculate covariance matrices
    cs = (s.unsqueeze(-1) * s.unsqueeze(-2)).mean(0).mean(0)
    cb = (b.unsqueeze(-1) * b.unsqueeze(-2)).mean(0).mean(0)
    cz = (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(0).mean(0)

    # Perform eigendecomposition and transform components
    d, v = torch.linalg.eigh(cs)
    ss = v.transpose(-2, -1) @ s.unsqueeze(-1)
    
    d, v = torch.linalg.eigh(cb)
    bb = v.transpose(-2, -1) @ b.unsqueeze(-1)
    
    d, v = torch.linalg.eigh(cz)
    zz = v.transpose(-2, -1) @ z.unsqueeze(-1)

    # Extract top principal components and normalize
    ss = ss.squeeze(-1)[..., -2:]
    bb = bb.squeeze(-1)[..., -2:]
    zz = zz.squeeze(-1)[..., -2:]

    ss = ss / ss.std()
    bb = bb / bb.std()
    zz = zz / zz.std()

    # Create visualization
    batch_num = 0
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Plot principal component scores
    axs[0].plot(zz[:, batch_num, -1:], 'r', label='s')
    axs[0].plot(bb[:, batch_num, -1:], 'g', label='b')
    axs[0].plot(ss[:, batch_num, -1:], 'b', label='z')
    axs[0].set_title('Top PC Scores')
    axs[0].legend()

    # Plot assignment probabilities
    axs[1].plot(p[:, batch_num, 2], 'r')
    axs[1].plot(p[:, batch_num, 1], 'g')
    axs[1].plot(p[:, batch_num, 0], 'b')
    axs[1].set_title('Number of Assigned Nodes')
    axs[1].set_xlabel('Time')
    plt.savefig('flame_pc_scores.png')
    plt.show()
