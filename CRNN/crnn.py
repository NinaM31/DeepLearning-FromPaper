import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, vocab_size, dropout=0.5):
        super(CRNN, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.dropout = nn.Dropout(dropout)

        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),

            nn.Conv2d(32, 64, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),

            nn.Conv2d(64, 128, (3,3), stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2), 2),

            nn.Conv2d(256, 512, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1,2), 2),

            nn.Conv2d(512, 512, (2,2), stride=1, padding=0),
            self.dropout
        )

        self.mapSeq = nn.Sequential(
            nn.Linear(6656, 64),
            self.dropout
        )
        
        self.lstm_0 = nn.LSTM(64, 256, bidirectional=True, num_layers=2, batch_first=True)  
        self.lstm_1 = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)

        self.out = nn.Linear(512, vocab_size)

        self.to(self.device)
        print('Model loaded to ', self.device)
        
        
    def forward(self, x): 
        x = x.to(self.device)
        x = self.convlayer(x)

        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)

        x, _ = self.lstm_0(x)
        x, _ = self.lstm_1(x)

        x = self.out(x)   
        return x