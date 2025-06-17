
"""
Newton's Cradle Example for Dynamic Markov Blanket Discovery.

This script demonstrates the use of DMBD on Newton's cradle simulation data,
analyzing collision dynamics and object interactions.
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
from simulations.NewtonsCradle import NewtonsCradle

start_time = time.time()

# Initialize Newton's cradle simulation
dmodel = NewtonsCradle(
    n_balls=5,
    ball_size=0.2,
    Tmax=500,
    batch_size=40,
    g=1,
    leak=0.05/8,
    dt=0.05,
    include_string=False
)

# Generate different types of data
data_temp = dmodel.generate_data('random')[0]
data_0 = data_temp[0::5]
data_temp = dmodel.generate_data('1 ball object')[0]
data_1 = data_temp[0::5]
data_temp = dmodel.generate_data('2 ball object')[0]
data_2 = data_temp[0::5]
# data_temp = dmodel.generate_data('3 ball object')[0]
# data_3 = data_temp[0::5]
# data_temp = dmodel.generate_data('4 ball object')[0]
# data_4 = data_temp[0::5]
data_temp = dmodel.generate_data('1 + 1 ball object')[0]
data_11 = data_temp[0::5]
data_temp = dmodel.generate_data('2 + 2 ball object')[0]
data_22 = data_temp[0::5]
data_temp = dmodel.generate_data('1 + 2 ball object')[0]
data_12 = data_temp[0::5]
# data_temp = dmodel.generate_data('2 + 3 ball object')[0]
# data_23 = data_temp[0::5]
# data_temp = dmodel.generate_data('1 + 3 ball object')[0]
# data_13 = data_temp[0::5]

datas = (data_0, data_1, data_2, data_11, data_12, data_22)
dy = torch.zeros(2)
delta = 0.5
xlim = (-1.5, 1.5)
ylim = (-1.2 + delta, 0.2 + delta)
dy[1] = delta
new_datas = ()
for k, data in enumerate(datas):
    data = data + dy
    v_data = torch.diff(data, dim=0)
    v_data = v_data / v_data.std()
    new_datas = new_datas + (torch.cat((data[1:], v_data), dim=-1),)
datas = new_datas


# num_mixtures = 5
# batch_shape = ()
# hidden_dims = (4,4,4)
# role_dims = (2,2,2)
# iters = 40
# model0 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model1 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model2 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model3 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model4 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model11 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model12 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model13 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model22 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)
# model23 = DMBD(obs_shape=(5,4),role_dims=role_dims,hidden_dims=hidden_dims,batch_shape=batch_shape,regression_dim = -1, control_dim=-1)

# models = []
# data = torch.cat(datas[0:3],dim=1)
# for i in range(10):
#     model = DMBD(obs_shape=data.shape[-2:],role_dims=(8,4,8),hidden_dims=(4,2,4),batch_shape=(),regression_dim = -1, control_dim=0)
#     models.append(model)

# ELBO = []
# for k, model in enumerate(models):
#     model.update(data,None,None,iters=20,latent_iters=1,lr=0.5)
#     ELBO.append(model.ELBO())

# Combine datasets for training
data = torch.cat(datas[3:5], dim=1).clone().detach()
data = torch.cat((datas[0], data), dim=1).clone().detach()
print('Simulations complete')

# Initialize DMBD model
model = DMBD(
    obs_shape=data.shape[-2:],
    role_dims=(8, 8, 8),
    hidden_dims=(4, 4, 4),
    batch_shape=(),
    regression_dim=-1,
    control_dim=0
)

# Training parameters
iters = 80
r1 = model.role_dims[0]
r2 = r1 + model.role_dims[1]
r3 = r2 + model.role_dims[2]
h1 = model.hidden_dims[0]
h2 = h1 + model.hidden_dims[1]
h3 = h2 + model.hidden_dims[2]
batch_num = 50

# Train the model with visualization
for i in range(iters):
    model.update(data, None, None, iters=1, latent_iters=1, lr=0.5, verbose=True)

    # Extract model predictions and roles
    sbz = model.px.mean()
    B = model.obs_model.obs_dist.mean()
    if model.regression_dim == 0:
        roles = B @ sbz
    else:
        roles = B[..., :-1] @ sbz + B[..., -1:]
    sbz = sbz.squeeze()
    roles = roles.squeeze()
    idx = model.obs_model.NA / model.obs_model.NA.sum() > 0.01

    r1 = model.role_dims[0]
    r2 = r1 + model.role_dims[1]
    r3 = r2 + model.role_dims[2]

    pbar = model.obs_model.NA / model.obs_model.NA.sum()
    pbar = pbar / pbar.max()
    p1 = model.obs_model.p[:, batch_num, :, list(range(0, r1))].mean(-2)
    p2 = model.obs_model.p[:, batch_num, :, list(range(r1, r2))].mean(-2)
    p3 = model.obs_model.p[:, batch_num, :, list(range(r2, r3))].mean(-2)

    plt.scatter(
        roles[:, batch_num, list(range(0, r1)), 0],
        roles[:, batch_num, list(range(0, r1)), 1],
        color='r',
        alpha=0.25
    )
    plt.scatter(
        roles[:, batch_num, list(range(r1, r2)), 0],
        roles[:, batch_num, list(range(r1, r2)), 1],
        color='g',
        alpha=0.25
    )
    plt.scatter(
        roles[:, batch_num, list(range(r2, r3)), 0],
        roles[:, batch_num, list(range(r2, r3)), 1],
        color='b',
        alpha=0.25
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

# Principal component analysis
p = model.assignment_pr()
p = p.sum(-2)
print('Show PC scores')
s = sbz[:, :, 0:h1]
s = s - s.mean(0).mean(0)
b = sbz[:, :, h1:h2]
b = b - b.mean(0).mean(0)
z = sbz[:, :, h2:h3]
z = z - z.mean(0).mean(0)

cs = (s.unsqueeze(-1) * s.unsqueeze(-2)).mean(0).mean(0)
cb = (b.unsqueeze(-1) * b.unsqueeze(-2)).mean(0).mean(0)
cz = (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(0).mean(0)

d, v = torch.linalg.eigh(cs)
ss = v.transpose(-2, -1) @ s.unsqueeze(-1)
print('Normalized Eigenvalues of s', d / d.sum())
d, v = torch.linalg.eigh(cb)
print('Normalized Eigenvalues of b', d / d.sum())
bb = v.transpose(-2, -1) @ b.unsqueeze(-1)
d, v = torch.linalg.eigh(cz)
print('Normalized Eigenvalues of z', d / d.sum())
zz = v.transpose(-2, -1) @ z.unsqueeze(-1)

ss = ss.squeeze(-1)[..., -2:]
bb = bb.squeeze(-1)[..., -2:]
zz = zz.squeeze(-1)[..., -2:]

ss = ss / ss.std()
bb = bb / bb.std()
zz = zz / zz.std()

# Plot principal component scores
fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(ss[:, batch_num, -1:], 'r', label='s')
axs[0].plot(bb[:, batch_num, -1:], 'g', label='b')
axs[0].plot(zz[:, batch_num, -1:], 'b', label='z')
axs[0].set_title('Top PC Score')
axs[0].legend()

axs[1].plot(p[:, batch_num, 0], 'r')
axs[1].plot(p[:, batch_num, 1], 'g')
axs[1].plot(p[:, batch_num, 2], 'b')
axs[1].set_title('Number of Assigned Objects')
axs[1].set_xlabel('Time')
plt.show()

# Generate animation
print('Generating Movie...')
f = r"./cradle.mp4"
ar = animate_results('sbz', f, xlim=xlim, ylim=ylim, fps=10)
ar.make_movie(model, data, (0, 20, 40, 60, 80, 100))

# Final visualization
batch_num = 40
plt.scatter(
    data[:, batch_num, :, 0],
    data[:, batch_num, :, 1],
    cmap='rainbow_r',
    c=model.obs_model.p.argmax(-1)[:, batch_num, :]
)
plt.show()

print(f"Script completed in {time.time() - start_time:.2f} seconds")
