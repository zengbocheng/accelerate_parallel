"""
@author: bochengz
@date: 2023/04/14
@email: bochengzeng@bochengz.top
"""
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
