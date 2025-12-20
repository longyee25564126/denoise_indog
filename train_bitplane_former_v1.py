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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from models.bitplane_former_v1 import BitPlaneFormerV1
from datasets.bitplane_utils import expand_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BitPlaneFormerV1 with residual-based mask supervision.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config; CLI args override this.")
    # Data
    parser.add_argument("--root", type=str, default=None, help="Dataset root.")
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
    parser.add_argument("--warmup-epochs", type=float, default=0.0, help="Linear warmup duration in epochs before cosine decay (converted to steps).")
    parser.add_argument("--lambda-mask", type=float, default=0.1, help="Weight for mask regression loss.")
    parser.add_argument("--mask-T", type=float, default=48.0, help="Temperature for residual mask normalization (dataset + training).")
    parser.add_argument("--mask-type", type=str, default="soft", choices=["soft", "binary"], help="Mask supervision type.")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Threshold when using binary mask supervision.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; 0 to disable.")
    parser.add_argument("--amp", action="store_true", help="Use AMP (autocast + GradScaler).")
    parser.add_argument("--save-dir", type=str, default="outputs/train_bitplane_former_v1")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between logs.")
    parser.add_argument("--device", type=str, default=None, help='Device (default: "cuda" if available else "cpu").')
    parser.add_argument("--compute-lpips", action="store_true", help="Compute LPIPS in validation (requires lpips package).")
    parser.add_argument("--dec-type", type=str, default="fuse_encoder", choices=["fuse_encoder", "decoder_q_msb", "std_encdec_msb"], help="Decoder variant.")
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


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute per-sample PSNR. pred/target: (B,3,H,W) or (3,H,W) in [0,1].
    Returns: tensor of shape (B,) or scalar if no batch dim.
    """
    if pred.dim() == 3:
        mse = F.mse_loss(pred, target)
        if mse.item() == 0:
            return torch.tensor(float("inf"))
        return torch.tensor(20 * math.log10(1.0) - 10 * math.log10(mse.item() + eps))

    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=1)  # (B,)
    psnr_vals = 20 * torch.log10(torch.tensor(1.0, device=pred.device)) - 10 * torch.log10(mse + eps)
    return psnr_vals


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, window_size: int = 11, window_sigma: float = 1.5) -> float:
    """
    Compute SSIM for a single image pair. pred/target: (3,H,W) in [0,1].
    """
    import math

    # create gaussian window
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * window_sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    window_1d = g
    window_2d = window_1d.transpose(1, 2) @ window_1d
    window = window_2d.expand(3, 1, window_size, window_size)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(pred.unsqueeze(0), window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(target.unsqueeze(0), window, padding=window_size // 2, groups=3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred.unsqueeze(0) * pred.unsqueeze(0), window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target.unsqueeze(0) * target.unsqueeze(0), window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred.unsqueeze(0) * target.unsqueeze(0), window, padding=window_size // 2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


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
    # Load YAML config if provided, then let CLI override
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if cfg:
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    if args.root is None:
        raise ValueError("Please specify --root or set it in the YAML config.")

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
        dec_type=args.dec_type,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(math.ceil(max(0.0, args.warmup_epochs) * len(train_loader)))

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine decay from 1 to 0
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    best_psnr = -1e9
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scaler)
        log_write(f"Resumed from {args.resume} at epoch {start_epoch}, best_psnr={best_psnr:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    # Truncate existing log on each run
    with open(log_path, "w") as _f:
        pass
    def log_write(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Save args to log at start
    log_write("==== Training Configuration ====")
    for k, v in sorted(vars(args).items()):
        log_write(f"{k}: {v}")
    log_write("================================")

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lsb = batch["lsb"].to(device)
            msb = batch["msb"].to(device)
            mask_gt = batch.get("mask_gt", None)
            if mask_gt is not None:
                mask_gt = mask_gt.to(device)
                if mask_gt.ndim == 3:
                    mask_gt = mask_gt.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                out = model({"x": x, "lsb": lsb, "msb": msb})
                loss_recon = F.l1_loss(out["y_hat"], y)
                if args.dec_type == "std_encdec_msb":
                    loss_mask = torch.tensor(0.0, device=device)
                    loss = loss_recon
                else:
                    if args.mask_type == "binary":
                        mask_gt_bin = (mask_gt >= args.mask_thresh).float()
                        loss_mask = F.binary_cross_entropy_with_logits(out["m_logits"], mask_gt_bin)
                    else:
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
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                with torch.no_grad():
                    if args.dec_type == "std_encdec_msb":
                        log_msg = (
                            f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                            f"loss {loss.item():.4f} (recon {loss_recon.item():.4f}) "
                            f"lr {optimizer.param_groups[0]['lr']:.6f}"
                        )
                    else:
                        m_hat = out["m_hat"]
                        log_msg = (
                            f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                            f"loss {loss.item():.4f} (recon {loss_recon.item():.4f}, mask {loss_mask.item():.4f}) "
                            f"mask_gt mean/std/min/max {mask_gt.mean().item():.4f}/{mask_gt.std().item():.4f}/"
                            f"{mask_gt.min().item():.4f}/{mask_gt.max().item():.4f} "
                            f"m_hat mean/std/min/max {m_hat.mean().item():.4f}/{m_hat.std().item():.4f}/"
                            f"{m_hat.min().item():.4f}/{m_hat.max().item():.4f} "
                            f"lr {optimizer.param_groups[0]['lr']:.6f}"
                        )
                    log_write(log_msg)

        epoch_time = time.time() - t0
        log_write(f"Epoch {epoch+1} done in {epoch_time:.1f}s, avg loss {epoch_loss/len(train_loader):.4f}")

        # Validation
        val_psnr = None
        val_ssim = None
        val_lpips = None
        if val_loader is not None:
            model.eval()
            psnr_acc = 0.0
            ssim_acc = 0.0
            lpips_acc = 0.0
            count = 0
            lpips_fn = None
            if args.compute_lpips:
                try:
                    import lpips
                    lpips_fn = lpips.LPIPS(net="vgg").to(device)
                except Exception as e:
                    print(f"LPIPS not computed (import failed: {e})")
                    lpips_fn = None
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    lsb = batch["lsb"].to(device)
                    msb = batch["msb"].to(device)
                    out = model({"x": x, "lsb": lsb, "msb": msb})
                    y_hat = out["y_hat"].clamp(0.0, 1.0)
                    psnr_vals = psnr(y_hat, y)
                    if psnr_vals.dim() == 0:
                        psnr_acc += psnr_vals.item()
                        count += 1
                    else:
                        psnr_acc += psnr_vals.sum().item()
                        count += psnr_vals.numel()
                    ssim_acc += ssim(y_hat[0], y[0]) if y_hat.size(0) == 1 else sum(
                        ssim(y_hat[b], y[b]) for b in range(y_hat.size(0))
                    )
                    if lpips_fn is not None:
                        y_hat_lp = (y_hat * 2 - 1).clamp(-1, 1)
                        y_lp = (y * 2 - 1).clamp(-1, 1)
                        lpips_acc += lpips_fn(y_hat_lp, y_lp).mean().item()
            val_psnr = psnr_acc / max(count, 1)
            val_ssim = ssim_acc / max(count, 1)
            if lpips_fn is not None:
                val_lpips = lpips_acc / max(count, 1)
            msg = f"Val PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}"
            if val_lpips is not None:
                msg += f" | LPIPS: {val_lpips:.4f}"
            log_write(msg)

        # Checkpointing
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "epoch": epoch + 1,
            "best_psnr": best_psnr,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips,
            "args": vars(args),
        }
        if val_psnr is not None and val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt["best_psnr"] = best_psnr
            save_checkpoint(ckpt, os.path.join(args.save_dir, "best.pth"))
            log_write(f"Saved new best checkpoint with PSNR {best_psnr:.3f}")
        ckpt["best_psnr"] = best_psnr
        save_checkpoint(ckpt, os.path.join(args.save_dir, "last.pth"))


if __name__ == "__main__":
    main()
