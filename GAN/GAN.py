import numpy as np
import pickle as pkl

import torch 
import torch.nn as nn

from Discriminator import Discriminator
from Generator import Generator


class GAN:
    
    def __init__(self, z_size, input_size, d_hidden_dim, g_hidden_dim, d_out_size, g_out_size):

        self.D = Discriminator(input_size, d_hidden_dim, d_out_size)
        self.G = Generator(z_size, g_hidden_dim, g_out_size)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
        self.D.to(self.device)
        self.G.to(self.device)

    
    def __calculate_loss(self, output, labels):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(output.squeeze(), labels)


    def real_loss(self, D_out):
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size).to(self.device)

        return self.__calculate_loss(D_out, labels) 

    
    def fake_loss(self, D_out):
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size).to(self.device)

        return self.__calculate_loss(D_out, labels) 


    def rescale_image(self, image):
        return image*2 - 1


    def describe(self):
        print('Discriminator')
        print(self.D)

        print('\nGenerator')
        print(self.G)


    def noise(self, size):
        z = np.random.uniform(-1, 1, size=size)
        return torch.from_numpy(z).float().to(self.device)

    def train_generator(self, g_optim, size):
        g_optim.zero_grad()

        z = self.noise(size)
        fake_images = self.G(z)

        d_fake = self.D(fake_images)
        g_loss = self.real_loss(d_fake)

        g_loss.backward()
        g_optim.step()

        return g_loss.item()

    
    def train_discriminator(self, d_optim, real_images, size):
        d_optim.zero_grad()

        d_real = self.D(real_images.to(self.device))
        d_real_loss = self.real_loss(d_real)

        z = self.noise(size)
        fake_images = self.G(z)

        d_fake = self.D(fake_images)
        d_fake_loss = self.fake_loss(d_fake)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optim.step()

        return d_loss.item()

    def train(self, num_epochs, d_optim, g_optim, data_loader, z_size, sample_size, print_every=500):
        samples, losses = [], []

        z = self.noise((sample_size, z_size))

        self.D.train()
        self.G.train()

        print(f'Running on {self.device}')
        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(data_loader):                    
                batch_size = real_images.size(0)
                real_images = self.rescale_image(real_images)

                d_loss = self.train_discriminator(d_optim, real_images, (sample_size, z_size))
                g_loss = self.train_generator(g_optim, (sample_size, z_size))

                if i % print_every == 0:
                    print('Epoch [{:5d}/{:5d}] | d_loss {:6.4f} | g_loss {:6.4f}'.format(
                        epoch+1,
                        num_epochs,
                        d_loss,
                        g_loss
                    ))

            losses.append( (d_loss, g_loss) )

            self.G.eval()
            samples.append( self.G(z) )
            self.G.train()

        with open('GAN_Sample_Output.pkl', 'wb') as f:
            pkl.dump(samples, f)

        return samples, losses