
"""
Flocking Behavior Example for Dynamic Markov Blanket Discovery.

This script demonstrates the use of DMBD on flocking simulation data,
analyzing collective behavior and role discovery.
"""

import os
import sys
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm

from models.DynamicMarkovBlanketDiscovery import DMBD, animate_results

start_time = time.time()

from simulations.Flocking import Flocking

flocking_sim = Flocking(n_birds=10, world_size=10.0)
flocking_sim.preset_murmuration()
trajectory = flocking_sim.simulate(n_steps=1000, batch_size=10)
data = trajectory[:, :, :, :2]  # Extract positions only

# Initialize DMBD model for flocking analysis
model = DMBD(
    obs_shape=data.shape[-2:],
    role_dims=(1, 2, 2),
    hidden_dims=(4, 4, 4),
    regression_dim=-1,
    control_dim=0,
    number_of_objects=3,
    unique_obs=False
)

# Train the model
model.update(data, None, None, iters=20, latent_iters=1, lr=1, verbose=True)

# Extract model predictions and roles
sbz = model.px.mean()
B = model.obs_model.obs_dist.mean()

if model.regression_dim == 1:
    roles = B[..., :-1] @ sbz + B[..., -1:]
else:
    roles = B @ sbz

sbz = sbz.squeeze(-3).squeeze(-1)
roles = roles.squeeze(-1)[..., 0:2]

# Visualization parameters
batch_num = 1
temp1 = data[:, batch_num, :, 0]
temp2 = data[:, batch_num, :, 1]
rtemp1 = roles[:, batch_num, :, 0]
rtemp2 = roles[:, batch_num, :, 1]


# # Plot environment and roles
# idx = (model.assignment()[:, batch_num, :] == 0)
# plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.5)

# ev_dim = model.role_dims[0]
# ob_dim = np.sum(model.role_dims[1:])

# for i in range(ev_dim):
#     idx = (model.obs_model.assignment()[:, batch_num, :] == i)
#     plt.scatter(rtemp1[:, i], rtemp2[:, i])

# plt.title('Environment + Roles')
# plt.show()

# # Color mapping for different roles
# ctemp = model.role_dims[1] * ('b',) + model.role_dims[2] * ('r',)

# # Plot individual objects
# for j in range(model.number_of_objects):
#     idx = (model.assignment()[:, batch_num, :] == 0)
#     plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.2)
    
#     for i in range(1 + 2 * j, 1 + 2 * (j + 1)):
#         idx = (model.assignment()[:, batch_num, :] == i)
#         plt.scatter(temp1[idx], temp2[idx])
    
#     plt.title(f'Object {j + 1} (yellow is environment)')
#     plt.show()

#     # Plot object roles
#     idx = (model.assignment()[:, batch_num, :] == 0)
#     plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.2)
#     k = 0
    
#     for i in range(ev_dim + ob_dim * j, ev_dim + ob_dim * (j + 1)):
#         idx = (model.obs_model.assignment()[:, batch_num, :] == i)
#         plt.scatter(rtemp1[:, i], rtemp2[:, i], color=ctemp[k])
#         k = k + 1
    
#     plt.title(f'Object {j + 1} roles')
#     plt.show()


# Create animation
print('Making Movie')
f = r"flock.mp4"
ar = animate_results(
    'particular',
    f,
    xlim=(-7, 7),
    ylim=(-7, 7),
    fps=20
).make_movie(model, data, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
print('Done')

print(f"Script completed in {time.time() - start_time:.2f} seconds")
