import torch
import torch.nn as nn

class BaseModel(nn.Module):
    ''' Model architecture based on Paper (Basic)'''
    
    def __init__(self, in_feature, out_feature):
        super(BaseModel, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(in_feature, 500),
            nn.ReLU(),
            nn.Linear(500, 450),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(450, 400),
            nn.ReLU(),
            nn.Linear(400, 350),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(350, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(25, out_feature),
        )

    def forward(self, x):
        return self.base(x)