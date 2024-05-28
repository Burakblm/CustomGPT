import torch
import numpy as np
import os
import argparse

from utils import get_tokenizer

print("importing tokenizer...")
tokenizer = get_tokenizer()

def parse_args():
    parser = argparse.ArgumentParser(description='Tokenization for model training.')
    parser.add_argument('--data_path', type=str, default='/data/data.txt', help='Data path for tokenization')
    parser.add_argument('--split_rate', type=float, default=0.9, help='Partition ratio for training and validation datasets.')
    return parser.parse_args()

def check_file_extension(file_path):
    """
    This function is used to check file extensions.
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in [".json", ".txt"]:
        print(f"Valid file extension detected. File extension: {file_extension}")
        return file_extension
    else:
        raise ValueError("Unexpected file type")

def save_tensor(tensor: torch.Tensor, save_name: str, directory_path: str):
    tensor = tensor.to(torch.int16)
    save_name = os.path.join(directory_path, f"{save_name}.pt")
    torch.save(tensor, save_name)
    print(save_name)

def data_prepare(data_path: str, split_rate: float):
    data_extension = check_file_extension(data_path)
    directory_path = os.path.dirname(data_path)

    if data_extension in ".txt":
        with open(data_path, "r") as f:
            data = f.read()
        print("Text data is being tokenized...")
        data = tokenizer.encode(data)
        print("Text data tokenization process is complete.")
        if isinstance(data, list):
            print("Tokenized data is list")
        else:
            print("Warning: Tokenized data is not a list.")
        print("The data is being converted to a Tensor of type torch.tensor.")
        data = torch.tensor(data, dtype=torch.int16)
        print("The data conversion to tensor is complete.")
        data_size = len(data)
        print(f"The total number of tokens: {data_size}")
        print("The data is being split into train_data and val_data.")
        train_data = data[:int(data_size * split_rate)]
        print(f"Train data size: {len(train_data)}")
        val_data = data[int(data_size * split_rate):]
        print(f"Validation data size: {len(val_data)}")

        save_tensor(train_data, "train", directory_path)
        save_tensor(val_data, "val", directory_path)

        print("The dataset has been converted to train.pt and val.pt tensors.")

def load_tensor(data_path: str = "train.pt"):
    data = torch.load(data_path)
    print("tensor loading")
    return data

if __name__ == "__main__":
    args = parse_args()
    data_prepare(args.data_path, args.split_rate)
