import sys
import os
import importlib.util
import torch
import numpy as np

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("external_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_range():
    # Paths from config.yaml
    module_path = "/home/youming/dataset_and_data_loader/data_loader.py"
    root_dir = "/home/youming/dataset_and_data_loader/dataset final version"
    
    print(f"Loading module from: {module_path}")
    try:
        module = load_module(module_path)
    except Exception as e:
        print(f"Failed to load module: {e}")
        return

    if not hasattr(module, "PairedImageDataset"):
        print("Module has no PairedImageDataset class")
        return

    PairedImageDataset = getattr(module, "PairedImageDataset")
    print("Instantiating dataset...")
    
    try:
        # Try with split argument first
        ds = PairedImageDataset(root_dir=root_dir, split="train", transform=None)
    except TypeError:
        print("Constructor failed with 'split', trying without...")
        try:
            ds = PairedImageDataset(root_dir=root_dir, transform=None)
        except Exception as e:
            print(f"Failed to instantiate dataset: {e}")
            return

    print(f"Dataset length: {len(ds)}")
    if len(ds) == 0:
        print("Dataset is empty.")
        return

    # Check first sample
    print("Checking first sample...")
    item = ds[0]
    
    # Unpack based on length
    if len(item) == 3:
        noisy, clean, label = item
        print("Format: (noisy, clean, label)")
    elif len(item) == 4:
        noisy, _, clean, label = item
        print("Format: (noisy, path, clean, label)")
    else:
        print(f"Unknown item format length: {len(item)}")
        return

    # Check types and ranges
    for name, img in [("Noisy", noisy), ("Clean", clean)]:
        if isinstance(img, torch.Tensor):
            min_val = img.min().item()
            max_val = img.max().item()
            dtype = img.dtype
            print(f"{name} Image - Type: {dtype}, Min: {min_val:.4f}, Max: {max_val:.4f}")
            
            if max_val > 1.0:
                print(f"WARNING: {name} image max value > 1.0. It seems to be in [0, 255] range.")
            else:
                print(f"OK: {name} image seems to be in [0, 1] range.")
        else:
            print(f"{name} is not a tensor: {type(img)}")

if __name__ == "__main__":
    check_range()
