"""
@author: bochengz
@date: 2023/04/14
@email: bochengzeng@bochengz.top
"""
from torch.utils.data import Dataset
import numpy as np


class ToyDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.data = np.random.randn(self.length, 10).astype(np.float32)
        self.label = np.random.randn(self.length, 1).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item], self.label[item]