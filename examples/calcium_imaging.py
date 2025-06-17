
"""
Calcium Imaging Example for Dynamic Markov Blanket Discovery.

This script demonstrates the use of DMBD on calcium imaging data,
analyzing neural activity patterns and cell interactions.
"""

import os
import sys
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm

from models.DynamicMarkovBlanketDiscovery import DMBD

start_time = time.time()

print('Test on Calcium Imaging data')

# Load and preprocess calcium imaging data
data = torch.tensor(np.load('../data/calciumForJeff.npy')).float().unsqueeze(-1)
data = data / data.std()
v_data = data.diff(dim=0, n=1)
v_data = v_data / v_data.std()
data = torch.cat((data[1:], v_data), dim=-1)
data = data[:3600]
data = data.reshape(12, 300, 41, 2).swapaxes(0, 1).clone().detach()

# Initialize DMBD model for calcium imaging analysis
model = DMBD(
    obs_shape=data.shape[-2:],
    role_dims=(1, 1, 0),
    hidden_dims=(4, 2, 0),
    batch_shape=(),
    regression_dim=-1,
    control_dim=0,
    number_of_objects=5
)

# Train the model
model.update(data, None, None, iters=50, lr=0.5, verbose=True)

# Visualize results
batch_num = 0
t = torch.arange(0, data.shape[0]).view((data.shape[0],) + (1,) * (data.ndim - 1)).expand(data.shape)
plt.scatter(
    t[:, batch_num, :, 0],
    data[:, batch_num, :, 0],
    c=model.particular_assignment()[:, batch_num, :]
)
plt.title('Calcium Imaging Data with Object Assignments')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

# Compute object-specific averages
dbar = torch.zeros(
    data.shape[0:2] + (model.number_of_objects + 1,),
    requires_grad=False
)
ass = model.particular_assignment()
for n in range(model.number_of_objects + 1):
    temp = (data * (ass == n).unsqueeze(-1)).sum(-2)[..., 0]
    temp = temp / temp.std()
    temp.unsqueeze(-1)
    dbar[:, :, n] = temp.clone().detach()

print(f"Script completed in {time.time() - start_time:.2f} seconds")
