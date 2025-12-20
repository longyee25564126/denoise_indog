import argparse
import os
import sys
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Add root to sys.path to allow imports from datasets
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from youming_models.external_adapter import ExternalPairedBitPlaneDataset
from youming_models.bitplane_former_v1 import BitPlaneFormerV1


def train_one_epoch(model, dl, optimizer, device, epoch, args):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(dl):
        if args.limit_batches and i >= args.limit_batches:
            break

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lsb = batch["lsb"].to(device)
        msb = batch["msb"].to(device)
        mask_gt = batch["mask_gt"].to(device)

        optimizer.zero_grad()
        out = model(x, lsb, msb)

        m_hat = out["m_hat"]
        # Ensure mask_gt shape matches m_hat
        if mask_gt.ndim != m_hat.ndim:
            B, _, h, w = m_hat.shape
            mask_gt = mask_gt.view(B, 1, h, w)

        l1 = nn.L1Loss()(out["y_hat"], y)
        mask_loss = nn.SmoothL1Loss()(m_hat, mask_gt)
        loss = l1 + args.lambda_mask * mask_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0:
            print(f"Epoch {epoch} [{i}/{len(dl)}] Loss: {loss.item():.4f} (L1: {l1.item():.4f}, Mask: {mask_loss.item():.4f})")

    avg_loss = total_loss / (i + 1)
    duration = time.time() - start_time
    print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}. Time: {duration:.2f}s")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Improved BitPlaneFormerV1")
    parser.add_argument("--root", type=str, default="/home/youming/dataset_and_data_loader/dataset final version", help="Dataset root")
    parser.add_argument(
        "--external-module",
        type=str,
        default="/home/youming/dataset_and_data_loader/data_loader.py",
        help="Path to external data_loader.py",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="youming_models/output/checkpoint")
    parser.add_argument("--lambda-mask", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--limit-batches", type=int, default=0, help="Limit batches per epoch for debugging")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    
    args = parser.parse_args()

    # Load config if specified
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Override args with config values
        for k, v in config.items():
            # Only override if the argument exists in argparse
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(f"Warning: Config key '{k}' not found in arguments.")
        print(f"Loaded configuration from {args.config}")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading dataset from {args.root} using {args.external_module}")
    ds = ExternalPairedBitPlaneDataset(
        module_path=args.external_module,
        root_dir=args.root,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        augment=True,
        split="train"
    )
    print(f"Dataset size: {len(ds)}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        embed_dim=256,
        num_heads=8,
        msb_depth=6,
        dec_depth=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, dl, optimizer, args.device, epoch, args)
        
        # Save checkpoint based on interval
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
