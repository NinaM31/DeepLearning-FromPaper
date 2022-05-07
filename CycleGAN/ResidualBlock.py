import torch
from torch import nn as nn


class ResidualBlock(nn.Module):
   
    def __init__(self, conv_dim=32):
        super(ResidualBlock, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
        )


    def forward(self, x):
        out_1 = self.conv_layer1(x)
        out_2 = x + self.conv_layer2(out_1)

        return out_2