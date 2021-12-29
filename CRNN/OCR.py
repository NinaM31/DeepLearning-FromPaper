import torch
import torch.nn as nn

from config import *
from crnn import CRNN


class OCR:
    
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.crnn = CRNN(VOCAB_SIZE).to(self.device)
        print('Model loaded to ', self.device)

        self.critertion = nn.CTCLoss(blank=0)

        self.char2idx, self.idx2char = self.char_idx()


    def char_idx(self):
        char2idx = {}
        idx2char = {}

        characters = CHARS.lower() + '-'
        for i, char in enumerate(characters):
            char2idx[char] = i + 1
            idx2char[i + 1] = char

        return char2idx, idx2char
        

    def encode(self, labels):
        length_per_label = [len(label) for label in labels]
        joined_label = ''.join(labels)

        joined_encoding = []
        for char in joined_label:
           joined_encoding.append(self.char2idx[char]) 

        return (torch.IntTensor(joined_encoding), torch.IntTensor(length_per_label) )


    def decode(self, logits):
        tokens = logits.softmax(2).argmax(2).squeeze(1)
        
        tokens = ''.join([self.idx2char[token] 
                        if token != 0  else '-' 
                        for token in tokens.numpy()])
        tokens = tokens.split('-')
        
        text = [char 
                for batch_token in tokens 
                for idx, char in enumerate(batch_token)
                if char != batch_token[idx-1] or len(batch_token) == 1]
        text = ''.join(text)
        
        return text
        

    def calculate_loss(self, logits, labels):
        encoded_labels, labels_len = self.encode(labels)

        logits_lens = torch.full(
            size=(logits.size(1),), 
            fill_value=logits.size(0), 
            dtype=torch.int32
        ).to(self.device)

        return self.critertion(
            logits.log_softmax(2), encoded_labels, 
            logits_lens, labels_len
        )


    def train_step(self, optimizer, images, labels):
        logits = self.predict(images)

        optimizer.zero_grad()
        loss = self.calculate_loss(logits, labels)
        loss.backward()
        optimizer.step()

        return logits, loss

    
    def val_step(self, images, labels):
        logits = self.predict(images)
        loss = self.calculate_loss(logits, labels)

        return logits, loss


    def predict(self, img):
        return self.crnn(img.to(self.device))


    def train(self, num_epochs, optimizer, train_loader, val_loader, print_every=2):
        train_losses, valid_losses = [], []

        for epoch in range(num_epochs):

            tot_train_loss = 0
            self.crnn.train()
            for i, (images, labels) in enumerate(train_loader):
                logits, train_loss = self.train_step(optimizer, images, labels)
                
                tot_train_loss += train_loss.item()

            with torch.no_grad():

                tot_val_loss = 0
                self.crnn.eval()
                for i, (images, labels) in enumerate(val_loader):
                    logits, val_loss = self.val_step(images, labels)

                    tot_val_loss += val_loss.item()
                
                train_loss = tot_train_loss / len(train_loader.dataset)
                val_loss = tot_val_loss / len(val_loader.dataset)

                train_losses.append(train_loss)
                valid_losses.append(val_loss)

            if epoch % print_every == 0:
                print('Epoch [{:5d}/{:5d}] | train loss {:6.4f} | val loss {:6.4f}'.format(
                    epoch+1,
                    num_epochs,
                    train_loss,
                    val_loss
                )) 

        return train_losses, valid_losses