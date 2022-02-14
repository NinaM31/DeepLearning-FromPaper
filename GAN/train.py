import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

from GAN import GAN
from config import *


# Dataset
transform = transforms.ToTensor()

data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Model
gan_model = GAN(Z_SIZE, INPUT_SIZE, D_HIDDEN_DIM, G_HIDDEN_DIM, D_OUT_SIZE, G_OUT_SIZE)

# Oprimizer
d_optimizer = optim.Adam(gan_model.D.parameters(), lr)
g_optimizer = optim.Adam(gan_model.G.parameters(), lr)

# train
sample_result, losses_history = gan_model.train(50, d_optimizer, g_optimizer, data_loader, Z_SIZE, SAMPLE_SIZE)

# Plot
fig, ax = plt.subplots()
losses = np.array(losses_history)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()

rows = 10
cols = 5
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(sample_result[::int(len(sample_result)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        img = img.cpu().detach()
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

plt.show()