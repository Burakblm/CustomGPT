import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random

class GPTDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int = 1024):
        self.data = data
        self.block_size = block_size
        self.n_samples = len(self.data) - self.block_size

    def __getitem__(self, index):
        start_index = index * self.block_size
        end_index = min(start_index + self.block_size, len(self.data))

        x = self.data[start_index:end_index]
        y = self.data[start_index + 1:end_index + 1]

        return x, y

    def __len__(self):
        return math.ceil(self.n_samples / self.block_size)

