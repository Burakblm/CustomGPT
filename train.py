import os
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import Transformer, ModelArgs

from utils import get_tokenizer
from prepare_data import load_tensor
from data_loader import GPTDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ddp = int(os.environ.get("RANK", -1)) != -1
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model.')
    parser.add_argument('--data_path', type=str, default="/data/", help="Folder containing training and validation data")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Epoch number for training')
    parser.add_argument('--block_size', type=int, default=1024, help='Block size for training')
    parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--num_samples_for_loss', type=int, default=20, help='How much data will be used to calculate the loss value?')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_iters', type=int, default=50, help='Number of evaluation iterations')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    return parser.parse_args()

args = parse_args()

data_path = args.data_path
batch_size = args.batch_size
block_size = args.block_size
eval_interval = args.eval_interval
learning_rate = args.learning_rate
eval_iters = args.eval_iters
dropout = args.dropout
epochs = args.epochs
num_samples_for_loss = args.num_samples_for_loss


def ddp_setup():
    init_process_group(backend="nccl")

def destroy_ddp():
    destroy_process_group()

if ddp:
    ddp_setup()

scaler = GradScaler()

def prepare_dataloader(dataset: Dataset, batch_size: int):
    if ddp:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset, shuffle=True)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=0
        )

def find_pt_files(directory_path: str):
    train_file_path = os.path.join(directory_path, 'train.pt')
    val_file_path = os.path.join(directory_path, 'val.pt')

    train_exists = os.path.isfile(train_file_path)
    val_exists = os.path.isfile(val_file_path)

    train_result = train_file_path if train_exists else None
    val_result = val_file_path if val_exists else None

    return train_result, val_result
        

def create_data_loader(data_path: str):
    data = load_tensor(data_path)
    dataset = GPTDataset(data, block_size=block_size)
    dataloader = prepare_dataloader(dataset, batch_size)
    return dataloader


train_data_path, val_data_path = find_pt_files(data_path)

train_dataloader = create_data_loader(train_data_path)
val_dataloader = create_data_loader(val_data_path)


model_args = ModelArgs()
model = Transformer(model_args)

if ddp:
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
else:
    model = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')


dataloader = {"train": train_dataloader,
              "val": val_dataloader}

@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if dataloader[split] is not None:
            losses = torch.zeros(num_samples_for_loss)
            loop = tqdm(enumerate(dataloader[split]), total=num_samples_for_loss, leave=True)
            for j, (inputs, targets) in loop:
                if j > num_samples_for_loss:
                    break
                inputs = inputs.to(gpu_id if ddp else device)
                targets = targets.to(gpu_id if ddp else device)
                logits, loss = model(inputs, targets)
                losses[j] = loss.item()
                loop.set_description(f"{split} Average Loss")
                loop.set_postfix(loss = loss.item())
        out[split] = losses.mean()
    model.train()
    return out


def train():
    for epoch in range(epochs):
        bs = len(next(iter(train_dataloader))[0])
        print(f"[GPU:{gpu_id if ddp else device}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(train_dataloader)}")
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)

        for i, (inputs, targets) in loop:
            if i % eval_interval == 0 and i > 0:
                if ddp:
                    if gpu_id == 0:
                        out = calculate_loss()
                        print(f"Train loss: {out['train']:.4f}" + (f" | Val loss : {out['val']:.4f}" if val_dataloader is not None else ""))
                else:
                    out = calculate_loss()
                    print(f"Train loss: {out['train']:.4f}" + (f" | Val loss : {out['val']:.4f}" if val_dataloader is not None else ""))

            inputs = inputs.to(gpu_id if ddp else device)
            targets = targets.to(gpu_id if ddp else device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits, loss = model(inputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss = loss.item())


if ddp:
    destroy_ddp()

if __name__ == "__main__":
    args = parse_args()
    train()