import torch

from DataPreperation import DataPreperation


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, check_point_file, is_checpoint_exist=False):

        with open(data_dir, encoding="utf-8") as f:
            data = f.readlines()

            processor = DataPreperation(data, check_point_file, is_checpoint_exist)
            self.X = processor.X
            self.Y = processor.Y


    def __len__(self):
        return len(self.Y)


    def __getitem__(self, idx):
        return torch.IntTensor(self.X[idx]), torch.IntTensor(self.Y[idx])