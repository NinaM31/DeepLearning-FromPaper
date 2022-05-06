import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from CycleGAN import CycleGAN
from Dataset import Dataset
from config import *


# Dataset
x_dataset = Dataset(X_DATASET)
y_dataset = Dataset(Y_DATASET)
test_x_dataset = Dataset(X_DATASET, data_type="test")
test_y_dataset = Dataset(Y_DATASET, data_type="test")

data_loader_x = DataLoader(x_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
data_loader_y = DataLoader(y_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
test_data_loader_x = DataLoader(test_x_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
test_data_loader_y = DataLoader(test_y_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

# Model
cycleGan = CycleGAN()

# Oprimizer
g_params = list(cycleGan.G_XtoY.parameters()) + list(cycleGan.G_YtoX.parameters())

optimizers = {
    "g_optim": optim.Adam(g_params, LR, [BETA1, BETA2]),
    "d_x_optim": optim.Adam(cycleGan.D_X.parameters(), LR, [BETA1, BETA2]),
    "d_y_optim": optim.Adam(cycleGan.D_Y.parameters(), LR, [BETA1, BETA2])
}

# Train
losses = cycleGan.train(optimizers, data_loader_x, data_loader_y, test_data_loader_x, test_data_loader_y)

# Plot
fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()