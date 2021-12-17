import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU(0.2)

        self.fc_hidden0 = self.fc_layer(input_size, hidden_dim*4)
        self.fc_hidden1 = self.fc_layer(hidden_dim*4, hidden_dim*2)
        self.fc_hidden2 = self.fc_layer(hidden_dim*2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_size)
    

    def fc_layer(self, input_dim, output_dim):
            
        return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                self.activation,
                self.dropout
            )


    def forward(self, x):

        x = self.fc_hidden0(x)
        x = self.fc_hidden1(x)
        x = self.fc_hidden2(x)

        return self.fc_out(x)