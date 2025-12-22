import argparse
import os
import random
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from datasets.bitplane_utils import to_uint8, expand_bits, make_lsb_msb
from models.bitplane_former_v1 import BitPlaneFormerV1


def load_dataset(args: argparse.Namespace) -> ExternalPairedBitPlaneDataset:
    import importlib.util

    spec = importlib.util.spec_from_file_location("external_dataloader", args.external_module)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {args.external_module}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PairedImageDataset"):
        raise AttributeError(f"Module {args.external_module} has no PairedImageDataset")
    PairedImageDataset = getattr(module, "PairedImageDataset")
    try:
        base_dataset = PairedImageDataset(root_dir=args.root, split=args.split, transform=None)
    except TypeError:
        base_dataset = PairedImageDataset(root_dir=args.root, transform=None)

    # We don't need the full BitPlaneDataset wrapper for just loading raw images, 
    # but it handles some logic nicely. However, we want to crop manually to 16x16 
    # AFTER picking the image, to ensure we get a valid random crop.
    # So let's just use the base dataset to get the full image, then crop.
    return base_dataset


def print_grid(name: str, tensor_u8: torch.Tensor, f=sys.stdout):
    """
    Print a 2D grid of values.
    tensor_u8: (H, W)
    """
    H, W = tensor_u8.shape
    print(f"--- {name} ({H}x{W}) ---", file=f)
    print("      " + " ".join(f"{c:3d}" for c in range(W)), file=f)
    for r in range(H):
        row_vals = " ".join(f"{v:3d}" for v in tensor_u8[r])
        print(f"Row{r:2d}: {row_vals}", file=f)
    print("", file=f)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a random 16x16 patch and dump pixel values.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root.")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--pad-size", type=int, default=0)
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--index", type=int, default=-1, help="Specific index to use, -1 for random.")
    parser.add_argument("--save-dir", type=str, default="outputs/patch_dump")
    parser.add_argument("--dump-file", type=str, default=None, help="File to save text output.")
    
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Load Model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # Align key settings with checkpoint when available.
    def _override_from_ckpt(name: str) -> None:
        if name in ckpt_args:
            new_val = ckpt_args[name]
            cur_val = getattr(args, name, None)
            if cur_val != new_val:
                print(f"Override {name}: {cur_val} -> {new_val} (from checkpoint)")
                setattr(args, name, new_val)

    _override_from_ckpt("patch_size")
    _override_from_ckpt("patch_stride")
    _override_from_ckpt("pad_size")
    _override_from_ckpt("lsb_bits")
    _override_from_ckpt("msb_bits")

    if args.patch_stride is None:
        args.patch_stride = args.patch_size

    # Allow CLI to override, but default to checkpoint values if not specified
    dec_type = ckpt_args.get("dec_type", "fuse_encoder")
    
    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        pad_size=args.pad_size,
        lsb_bits=expand_bits(args.lsb_bits),
        msb_bits=expand_bits(args.msb_bits),
        dec_type=dec_type,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load Dataset
    base_ds = load_dataset(args)
    if args.index < 0:
        idx = random.randint(0, len(base_ds) - 1)
    else:
        idx = args.index
    
    print(f"Processing sample index: {idx}")
    
    # Get full image pair
    # base_ds[idx] returns (noisy, clean, label)
    # noisy, clean are tensors (3, H, W) in [0, 1]
    noisy_full, clean_full, _ = base_ds[idx]
    
    # Random 16x16 crop
    _, H, W = noisy_full.shape
    crop_h, crop_w = 16, 16
    
    if H < crop_h or W < crop_w:
        raise ValueError(f"Image too small ({H}x{W}) for 16x16 crop")
        
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    
    noisy_patch = noisy_full[:, top:top+crop_h, left:left+crop_w]
    clean_patch = clean_full[:, top:top+crop_h, left:left+crop_w]
    
    # Prepare for model
    # Model expects batch dim
    x = noisy_patch.unsqueeze(0).to(device) # (1, 3, 16, 16)
    
    # Generate LSB/MSB
    x_u8_full = to_uint8(noisy_patch)
    lsb, msb = make_lsb_msb(x_u8_full, args.lsb_bits, args.msb_bits)
    lsb = lsb.unsqueeze(0).to(device)
    msb = msb.unsqueeze(0).to(device)
    
    # Run Inference
    with torch.no_grad():
        out = model({"x": x, "lsb": lsb, "msb": msb})
        y_hat = out["y_hat"].clamp(0, 1)
        m_hat = out.get("m_hat")
    
    # Convert to uint8 for display
    clean_u8 = to_uint8(clean_patch) # (3, 16, 16)
    denoise_u8 = to_uint8(y_hat.squeeze(0).cpu()) # (3, 16, 16)
    
    if m_hat is not None:
        # Mask is usually (1, 1, h, w) or (1, 1, H, W) depending on architecture
        # If it's pooled, we might need to interpolate or just show the small grid
        # But user asked for "pred mask", usually matching the image spatial dims if possible
        # or the raw output. Let's show the raw output first.
        # If m_hat is (1, 1, h, w), let's scale it to 0-255
        m_hat_val = m_hat.squeeze(0).cpu() # (1, h, w)
        # Interpolate to 16x16 if it's smaller (e.g. 2x2 or 4x4 tokens)
        if m_hat_val.shape[-1] != 16:
             m_hat_val = F.interpolate(m_hat_val.unsqueeze(0), size=(16, 16), mode='nearest').squeeze(0)
        
        mask_u8 = (m_hat_val.clamp(0, 1) * 255).to(torch.uint8)
    else:
        mask_u8 = None

    # Output Text
    out_stream = sys.stdout
    if args.dump_file:
        dump_dir = os.path.dirname(args.dump_file)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        out_stream = open(args.dump_file, "w")
        
    try:
        print(f"Sample Index: {idx}", file=out_stream)
        print(f"Crop: Top={top}, Left={left}, Size=16x16", file=out_stream)
        
        for c_idx, c_name in enumerate(["Red", "Green", "Blue"]):
            print_grid(f"Clean {c_name}", clean_u8[c_idx], f=out_stream)
            print_grid(f"Denoise {c_name}", denoise_u8[c_idx], f=out_stream)
            
        if mask_u8 is not None:
            print_grid("Pred Mask", mask_u8[0], f=out_stream)
        else:
            print("No mask predicted.", file=out_stream)
            
    finally:
        if args.dump_file and out_stream != sys.stdout:
            out_stream.close()
            print(f"Text output saved to {args.dump_file}")

    # Save Images
    os.makedirs(args.save_dir, exist_ok=True)
    
    def save_zoom(tensor_u8, name, scale=10):
        # tensor_u8: (C, H, W)
        arr = tensor_u8.permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
            mode = "L"
        else:
            mode = "RGB"
        img = Image.fromarray(arr, mode=mode)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        path = os.path.join(args.save_dir, f"{name}.png")
        img.save(path)
        print(f"Saved {path}")

    save_zoom(clean_u8, "clean_16x16")
    save_zoom(denoise_u8, "denoise_16x16")
    if mask_u8 is not None:
        save_zoom(mask_u8, "mask_16x16")
    
    # Also save noisy for reference
    save_zoom(to_uint8(noisy_patch), "noisy_16x16")

if __name__ == "__main__":
    main()
