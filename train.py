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
    parser.add_argument('--data_path', type=str, default=None, help="Folder containing training and validation data")
    parser.add_argument('--model_path', type=str, default=None , help='directory where the model will be saved')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Epoch number for training')
    parser.add_argument('--block_size', type=int, default=1024, help='Block size for training')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--num_samples_for_loss', type=int, default=100, help='How much data will be used to calculate the loss value?')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_iters', type=int, default=50, help='Number of evaluation iterations')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    return parser.parse_args()

args = parse_args()

data_path = args.data_path
model_path = args.model_path
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

if args.model_path is not None:
    model_path = args.model_path
else:
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "snapshot.pt")


model_args = ModelArgs()
model = Transformer(model_args)


if os.path.exists(model_path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        print("Loading from model file path...")
        model.module.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Loading from model file path...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

if ddp:
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
else:
    model = model.to(device)


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
            shuffle=True,
            num_workers=0
        )
    
class PTFileNotFoundError(Exception):
    """Custom exception raised when either or both of the files train.pt or val.pt are not found."""
    pass

def find_pt_files(directory_path: str):
    """
    Finds the train.pt and val.pt files in the specified directory.

    Args:
        directory_path (str): The directory path where the files will be searched.

    Returns:
        tuple: A tuple containing the paths of the found files (train.pt, val.pt).
               Returns None if the file is not found.

    Raises:
        PTFileNotFoundError: Raised if either of the files train.pt or val.pt is not found.
    """
    train_file_path = os.path.join(directory_path, 'train.pt')
    val_file_path = os.path.join(directory_path, 'val.pt')

    train_exists = os.path.isfile(train_file_path)
    val_exists = os.path.isfile(val_file_path)

    train_result = train_file_path if train_exists else None
    val_result = val_file_path if val_exists else None

    if not train_exists or not val_exists:
        missing_files = []
        if not train_exists:
            missing_files.append('train.pt')
        if not val_exists:
            missing_files.append('val.pt')
        raise PTFileNotFoundError(f"Files not found: {', '.join(missing_files)}")

    return train_result, val_result
        

def create_data_loader(data_path: str):
    data = load_tensor(data_path)
    dataset = GPTDataset(data, block_size=block_size)
    dataloader = prepare_dataloader(dataset, batch_size)
    return dataloader

try:
    train_data_path, val_data_path = find_pt_files(data_path)
    print("Train file path:", train_data_path)
    print("Val file path:", val_data_path)
except PTFileNotFoundError as e:
    print(e)

train_dataloader = create_data_loader(train_data_path)
val_dataloader = create_data_loader(val_data_path)


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
            loop = tqdm(enumerate(dataloader[split]), total=num_samples_for_loss-1, leave=True)
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
            if i % eval_interval == 0 and 1 == 0:#deneme için 0 yapıldı
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
            if gpu_id == 0:
                print(f"Model is being saved to the path {model_path} ...")
                torch.save(model.module.state_dict() if ddp else model.state_dict(), model_path)
                print(f"Epoch {epoch} | training snapshot save at snapshot.pt\n")
        else:
            print(f"Model is being saved to the path {model_path} ...")
            torch.save(model.module.state_dict() if ddp else model.state_dict(), model_path)
            print(f"Epoch {epoch} | training snapshot save at snapshot.pt\n")


if ddp:
    destroy_ddp()

if __name__ == "__main__":
    args = parse_args()
    train()