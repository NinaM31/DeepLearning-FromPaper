import yaml
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from trainer.FFNN_model_trainer import FFNNModelTrainer 
from Dataset import Dataset
from constants import *


# Load config
with open("Arabic_Diacritization/configs/FNN_basic.yml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Dataset
train_dataset = Dataset(
    TRAIN_DIR, 
    check_point_file=config["TRAIN_CHECK_POINT"],
    # is_checpoint_exist=True
)
val_dataset = Dataset(
    VAL_DIR, 
    check_point_file=config["VALID_CHECK_POINT"],
    # is_checpoint_exist=True
)

# DataLoaders
train_loader = DataLoader(
    train_dataset, 
    shuffle=True,
    batch_size=config["BATCH_SIZE"], 
    num_workers=config["N_WORKERS"],   
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False, 
    batch_size=config["BATCH_SIZE"], 
    num_workers=config["N_WORKERS"], 
)

# Model
fnn_basic = FFNNModelTrainer(config)

# Optimizer
optimizer = optim.Adagrad(fnn_basic.model, lr=config["lR"])

# train
train_losses, val_losses = fnn_basic.train(optimizer, train_loader, val_loader, print_every=1)

# Plot
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Valid Loss')
plt.title('Loss stats')
plt.legend()
plt.show()