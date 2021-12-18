import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

from GAN import GAN


# Dataset
num_workers = 0
batch_size = 64

transform = transforms.ToTensor()

data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers)

# Hyperparameters
z_size = 100
sample_size = 16

input_size = 28*28

d_hidden_dim = 32
d_out_size = 1

g_hidden_dim =  32
g_out_size = input_size

# Model
gan_model = GAN(z_size, input_size, d_hidden_dim, g_hidden_dim, d_out_size, g_out_size)

# Oprimizer
lr = 0.002
d_optimizer = optim.Adam(gan_model.D.parameters(), lr)
g_optimizer = optim.Adam(gan_model.G.parameters(), lr)

# train
sample_result, losses_history = gan_model.train(500, d_optimizer, g_optimizer, data_loader, z_size, sample_size)

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