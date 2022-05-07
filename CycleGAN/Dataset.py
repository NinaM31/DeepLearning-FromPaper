import os

import torch
from torchvision import transforms

from PIL import Image
from config import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        img_dir = BASE_DATASET_PATH + "/" + img_dir + "/"
        
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir) 

        self.img_dir = img_dir
        self.img_list = [os.path.join(abspath, path) for path in path_list]

        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # normalize image between -1 and 1
        ])


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = Image.open(path).convert('RGB')

        img_tensor = self.transform(img)
        return img_tensor