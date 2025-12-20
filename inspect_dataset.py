import sys
import os
import torch

# Add external module path
sys.path.append("/home/youming/dataset_and_data_loader")
from data_loader import PairedImageDataset

ds = PairedImageDataset(root_dir="/home/youming/dataset_and_data_loader/dataset final version")
print(f"Dataset length: {len(ds)}")
item = ds[0]
print(f"Item type: {type(item)}")
print(f"Item length: {len(item)}")
print(f"Item content types: {[type(x) for x in item]}")
if isinstance(item[0], torch.Tensor):
    print(f"Item 0 shape: {item[0].shape}")
if isinstance(item[1], torch.Tensor):
    print(f"Item 1 shape: {item[1].shape}")
if isinstance(item[0], torch.Tensor) and isinstance(item[2], torch.Tensor):
    diff = (item[0] - item[2]).abs().mean()
    print(f"L1 diff between item 0 and item 2: {diff}")

if isinstance(item[0], torch.Tensor) and isinstance(item[1], torch.Tensor):
    if item[1].ndim == 4:
        diff = (item[0] - item[1][0]).abs().mean()
        print(f"L1 diff between item 0 and item 1[0]: {diff}")

