import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        self.conv1 = self.conv_layer(3, conv_dim, batch_norm=False)
        self.conv2 = self.conv_layer(conv_dim, conv_dim*2,)
        self.conv3 = self.conv_layer(conv_dim*2, conv_dim*4,)
        
        self.fc = nn.Linear(conv_dim*4*4*4, 1)


    def conv_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        layers.append(conv_layer)

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=0.2))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, self.conv_dim*4*4*4)

        return self.fc(x)        