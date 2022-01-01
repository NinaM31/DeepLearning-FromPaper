import torch
from torch import nn as nn


class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*8)
        
        self.tconv1 = self.deconv_layer(conv_dim*8, conv_dim*8, stride=1, padding=0)
        self.tconv2 = self.deconv_layer(conv_dim*8, conv_dim*4)
        self.tconv3 = self.deconv_layer(conv_dim*4, conv_dim*2)
        self.tconv4 = self.deconv_layer(conv_dim*2, conv_dim)

        self.tconv5 = self.deconv_layer(conv_dim, 3, batch_norm=False, activation=False)
        self.dropout = nn.Dropout(0.25)

    
    def deconv_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, activation=True):
        layers = []

        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, self.conv_dim*8, 1, 1) 
        
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
      
        x = self.dropout(x)
            
        x = self.tconv5(x) 
        return torch.tanh(x)