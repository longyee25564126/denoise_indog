import argparse
import math
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from models.bitplane_former_v1 import BitPlaneFormerV1
from datasets.bitplane_utils import to_uint8, expand_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BitPlaneFormerV1 with residual-based mask supervision.")
    # Data
    parser.add_argument("--root", type=str, required=True, help="Dataset root.")
    parser.add_argument("--train-split", type=str, default="train", help="Split name for training.")
    parser.add_argument("--val-split", type=str, default=None, help="Split name for validation (optional).")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py", help="Path to data_loader.py defining PairedImageDataset.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--augment", action="store_true", help="Enable flip/rot90 augmentation for training.")
    parser.add_argument("--fit-to-patch", action="store_true", help="Center-crop to nearest patch-aligned size.")
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="LSB bit indices (default 0-5).")
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7], help="MSB bit indices (default 6-7).")
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-mask", type=float, default=0.1, help="Weight for mask regression loss.")
    parser.add_argument("--mask-T", type=float, default=48.0, help="Temperature for residual mask normalization.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; 0 to disable.")
    parser.add_argument("--amp", action="store_true", help="Use AMP (autocast + GradScaler).")
    parser.add_argument("--save-dir", type=str, default="outputs/train_bitplane_former_v1")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between logs.")
    parser.add_argument("--device", type=str, default=None, help='Device (default: "cuda" if available else "cpu").')
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace, split: str, train: bool) -> DataLoader:
    # Load external PairedImageDataset
    spec_transform = transforms.ToTensor()
    base_module_path = args.external_module
    import importlib.util

    spec = importlib.util.spec_from_file_location("external_dataloader", base_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {base_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PairedImageDataset"):
        raise AttributeError(f"Module {base_module_path} has no PairedImageDataset")
    PairedImageDatasetExt = getattr(module, "PairedImageDataset")
    try:
        base_dataset = PairedImageDatasetExt(root_dir=args.root, split=split, transform=spec_transform)
    except TypeError:
        base_dataset = PairedImageDatasetExt(root_dir=args.root, transform=spec_transform)

    ds = ExternalPairedBitPlaneDataset(
        base_dataset=base_dataset,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        augment=args.augment if train else False,
        return_mask_flat=False,
        split=split,
        fit_to_patch=args.fit_to_patch,
        use_residual_mask=True,
        mask_temperature=args.mask_T,
        mask_use_quantile=False,
        mask_quantile=0.9,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dl


def compute_mask_gt(x: torch.Tensor, y: torch.Tensor, patch_size: int, temperature: float) -> torch.Tensor:
    with torch.no_grad():
        x_u8 = to_uint8(x)
        y_u8 = to_uint8(y)
        err = (x_u8.to(torch.float32) - y_u8.to(torch.float32)).abs().mean(dim=1, keepdim=True)
        mask_pix = torch.clamp(err / temperature, 0.0, 1.0)
        mask_gt = F.avg_pool2d(mask_pix, kernel_size=patch_size, stride=patch_size)
    return mask_gt


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item() + eps)


def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: AdamW, scaler: GradScaler) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0)
    best_psnr = ckpt.get("best_psnr", -1e9)
    return start_epoch, best_psnr


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    lsb_bits = expand_bits(args.lsb_bits)
    msb_bits = expand_bits(args.msb_bits)

    # normalize bits
    args.lsb_bits = lsb_bits
    args.msb_bits = msb_bits

    train_loader = build_dataloader(args, args.train_split, train=True)
    val_loader = build_dataloader(args, args.val_split, train=False) if args.val_split else None

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        lsb_bits=lsb_bits,
        msb_bits=msb_bits,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_psnr = -1e9
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_psnr={best_psnr:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lsb = batch["lsb"].to(device)
            msb = batch["msb"].to(device)

            mask_gt = compute_mask_gt(x, y, args.patch_size, args.mask_T).to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                out = model({"x": x, "lsb": lsb, "msb": msb})
                loss_recon = F.l1_loss(out["y_hat"], y)
                loss_mask = F.smooth_l1_loss(out["m_hat"], mask_gt)
                loss = loss_recon + args.lambda_mask * loss_mask

            if args.amp:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                with torch.no_grad():
                    m_hat = out["m_hat"]
                    log_msg = (
                        f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                        f"loss {loss.item():.4f} (recon {loss_recon.item():.4f}, mask {loss_mask.item():.4f}) "
                        f"mask_gt mean/std {mask_gt.mean().item():.4f}/{mask_gt.std().item():.4f} "
                        f"m_hat mean/std {m_hat.mean().item():.4f}/{m_hat.std().item():.4f} "
                        f"lr {optimizer.param_groups[0]['lr']:.6f}"
                    )
                    print(log_msg, flush=True)

        scheduler.step()
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1} done in {epoch_time:.1f}s, avg loss {epoch_loss/len(train_loader):.4f}")

        # Validation
        val_psnr = None
        if val_loader is not None:
            model.eval()
            psnr_acc = 0.0
            count = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    lsb = batch["lsb"].to(device)
                    msb = batch["msb"].to(device)
                    out = model({"x": x, "lsb": lsb, "msb": msb})
                    y_hat = out["y_hat"].clamp(0.0, 1.0)
                    psnr_acc += psnr(y_hat, y)
                    count += 1
            val_psnr = psnr_acc / max(count, 1)
            print(f"Val PSNR: {val_psnr:.3f}")

        # Checkpointing
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "epoch": epoch + 1,
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        if val_psnr is not None and val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt["best_psnr"] = best_psnr
            save_checkpoint(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"Saved new best checkpoint with PSNR {best_psnr:.3f}")
        ckpt["best_psnr"] = best_psnr
        save_checkpoint(ckpt, os.path.join(args.save_dir, "last.pth"))


if __name__ == "__main__":
    main()
