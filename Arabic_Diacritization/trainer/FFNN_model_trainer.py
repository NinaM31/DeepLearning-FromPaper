import yaml
import torch 
import torch.nn as nn

from FFNN_model.BaseModel import BaseModel
from constants import *


class FFNNModelTrainer:
    ''' Trainer for all FFNN models '''

    def __init__(self, config):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.get_model(config).to(self.device)
        self.char2idx, self.idx2char = CHAR_IDX()
        self.config = config
        self.critertion = nn.CrossEntropyLoss()

        print('Model loaded to ', self.device)


    def calculate_loss(self, preds, Y_batch):
        return self.critertion(pred, Y_batch.to(self.device))


    def predict(self, X_batch):
        return self.model(X_batch.to(self.device))


    def train_step(self, optimizer, X_batch, Y_batch):
        preds = self.predict(X_batch)
        
        optimizer.zero_grad()
        loss = self.calculate_loss(preds, Y_batch)
        loss.backward()
        optimizer.step()

        return loss


    def val_step(self, optimizer, X_batch, Y_batch):
        preds = self.predict(X_batch)
        loss = self.calculate_loss(preds, Y_batch)

        return loss
    

    def train(self, optimizer, train_loader, val_loader, print_every=100):
        train_losses, valid_losses = [], []
        val_loss_min = np.Inf

        num_epochs = self.config["EPOCHS"]
        file_name = f"{BASE_PATH}/{self.config['SAVE_AS']}.pt"

        for epoch in range( num_epochs ):
            
            tot_train_loss = 0

            self.model.train()
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                train_loss = self.train_step(optimizer, X_batch, Y_batch)
                tot_train_loss += train_loss.item()

            with torch.no_grad():
                tot_val_loss = 0

                self.model.eval()
                for i, (X_batch, Y_batch) in enumerate(val_loader):
                    val_loss = self.val_step(optimizer, X_batch, Y_batch)
                    tot_val_loss += val_loss.item()

            # calculate loss
            train_loss = tot_train_loss / len(train_loader.dataset)
            val_loss = tot_val_loss / len(val_loader.dataset)

            train_losses.append(train_loss)
            valid_losses.append(val_loss)
            
            # show progress
            if epoch % print_every == 0:
                print('Epoch [{:5d}/{:5d}] | train loss {:6.4f} | val loss {:6.4f}'.format(
                    epoch+1    , num_epochs,
                    train_loss , val_loss
                )) 
            
            # save model
            if val_loss <= val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    val_loss_min, val_loss
                ))

                torch.save(self.model.state_dict(), file_name)
                val_loss_min = val_loss


    def get_model(self, config):
        if config['MODAL'] == "BaseModal":
            return BaseModel(config['FFNN_IN'], config['FFNN_OUT'])
        else:
            raise ValueError('Modal Not Supported')