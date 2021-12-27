import os

import torch 
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)

        self.img_dir = img_dir
        self.img_list = [os.path.join(abspath, path) for path in path_list]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        path = self.img_list[idx]

        img_name = os.path.basename(path).split('.')[0].lower().strip()
        img = Image.open(path).convert('RGB')

        img_tensor = self.transform(img)
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, img_name