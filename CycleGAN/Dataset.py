import os

import torch
from torchvision import transforms

from PIL import Image
from config import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, data_type="train"):
        img_dir = BASE_DATASET_PATH + "/" + img_dir + "/"
        
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir) 

        if data_type == "train": 
            start = 0
            end = round(len(path_list)*0.7)
        else: 
            start = round(len(path_list)*0.7) + 1
            end = len(path_list)

        self.img_dir = img_dir
        self.img_list = [os.path.join(abspath, path) for path in path_list[start: end]]

        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = Image.open(path).convert('RGB')

        img_tensor = self.transform(img)
        return img_tensor