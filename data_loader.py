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


"""
# Örnek veri oluşturalım
data = torch.arange(100000)  # 0'dan 99999'a kadar olan sayılar

# Veri setini oluşturalım
dataset = GPTDataset(data, block_size=1024)

print(dataset.__len__())


# DataLoader oluşturalım (shuffle=True ile her epoch'ta veri kümesini karıştır)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Veri yükleme işlemi
for epoch in range(2):
    print(f"Epoch {epoch+1}:")
    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"Batch {batch_idx}:\n")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        print(f"X:\n{x}")
        print(f"Y:\n{y}\n")

"""
