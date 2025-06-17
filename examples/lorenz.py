
"""
Lorenz Attractor Example for Dynamic Markov Blanket Discovery.

This script demonstrates the use of DMBD on the Lorenz attractor dataset,
performing dynamic discovery and visualization of results in 2D and 3D.
"""

import os
import sys

import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize

from models.DynamicMarkovBlanketDiscovery import DMBD
from simulations import Lorenz

print('Test on Lorenz attractor')

# Setup colormap for visualization
cmap = ListedColormap(['red', 'green', 'blue'])
vmin = 0  # Minimum value of the color scale
vmax = 2  # Maximum value of the color scale
norm = Normalize(vmin=vmin, vmax=vmax)

# Generate Lorenz attractor data
sim = Lorenz()
data = sim.simulate(100)

# Reshape and normalize data
data = torch.cat(
    (data[..., 0, :], data[..., 1, :], data[..., 2, :]),
    dim=-1
).unsqueeze(-2)
data = data - data.mean((0, 1, 2), True)
data = data / data.std()

# Initialize DMBD model
model = DMBD(
    obs_shape=data.shape[-2:],
    role_dims=(4, 4, 4),
    hidden_dims=(3, 3, 3),
    batch_shape=(),
    regression_dim=0,
    control_dim=0,
    number_of_objects=1
)
model.obs_model.ptemp = 6.0

# Training parameters
iters = 10
loc1 = torch.tensor((-0.5, -0.6, 1.6))
loc2 = torch.tensor((0.5, 0.6, 1.6))
# Training loop with visualization
for i in range(iters):
    # Update model with data
    model.update(data, None, None, iters=2, latent_iters=1, lr=0.5, verbose=True)

    # Get model predictions and calculate dimensions
    sbz = model.px.mean().squeeze()
    r1 = model.role_dims[0]
    r2 = r1 + model.role_dims[1]
    r3 = r2 + model.role_dims[2]
    h1 = model.hidden_dims[0]
    h2 = h1 + model.hidden_dims[1]
    h3 = h2 + model.hidden_dims[2]

    # Update colormap for assignments
    cmap = ListedColormap(['blue', 'green', 'red'])

    # Get assignment probabilities and assignments
    p = model.assignment_pr()
    a = model.assignment()
    batch_num = 0

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data[:, batch_num, 0, 0],
        data[:, batch_num, 0, 2],
        data[:, batch_num, 0, 4],
        cmap=cmap,
        norm=norm,
        c=a[:, batch_num, 0]
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('lorenz3d.png')
    plt.show()

    # Create 2D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        data[:, batch_num, :, 0],
        data[:, batch_num, :, 4],
        c=a[:, batch_num, :],
        cmap=cmap,
        norm=norm
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.savefig('lorenz2d.png')
    plt.show()

    # Distance-based scatter plot
    d1 = (data[..., 0::2] - loc1).pow(2).sum(-1).sqrt()
    d2 = (data[..., 0::2] - loc2).pow(2).sum(-1).sqrt()

    plt.scatter(
        d1[:, batch_num],
        d2[:, batch_num],
        c=a[:, batch_num, :],
        cmap=cmap,
        norm=norm
    )
    plt.show()

# Principal Component Analysis
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

# Create final visualization
batch_num = 0
fig, axs = plt.subplots(2, 1, sharex=True)

# Plot principal component scores
axs[0].plot(zz[:, batch_num, -1:], 'r', label='s')
axs[0].plot(bb[:, batch_num, -1:], 'g', label='b')
axs[0].plot(ss[:, batch_num, -1:], 'b', label='z')
axs[0].set_title('Top PC Score')
axs[0].legend()

# Plot assignment probabilities
axs[1].plot(p[:, batch_num, 2], 'r')
axs[1].plot(p[:, batch_num, 1], 'g')
axs[1].plot(p[:, batch_num, 0], 'b')
axs[1].set_title('Number of Assigned Nodes')
axs[1].set_xlabel('Time')
plt.savefig('lorenz_pc_scores.png')
plt.show()


