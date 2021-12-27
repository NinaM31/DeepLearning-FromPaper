import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, vocab_size, dropout=0.5):
        super(CRNN, self).__init__()

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

            nn.Conv2d(512, 512, (2,2), stride=1, padding=0)
        )

        self.mapSeq = nn.Linear(1024, 64)

        self.lstmlayer = nn.Sequential(
            nn.LSTM(64, 256, bidirectional=True, num_layers=2, batch_first=True),
            nn.LSTM(256, 256, bidirectional=True, num_layers=2, batch_first=True),
        )

        self.out = nn.Linear(256*2, vocab_size)
        
        
    def forward(self, x): 
        x = self.convlayer(x)
        x = self.dropout(x)
        x = self.mapSeq(x)
        x = self.lstmlayer(x)

        return self.out(x)