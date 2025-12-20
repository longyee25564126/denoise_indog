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
from youming_models.unet_model import BitPlaneUNet
from youming_models.metrics import calculate_psnr, calculate_ssim, LPIPS
from datasets.bitplane_utils import residual_soft_mask
from PIL import Image

def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    img: float tensor in [0,1], shape (3,H,W) or (1,H,W)
    """
    if img.dim() == 3 and img.shape[0] == 3:
        arr = (img.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr, mode="RGB")
    elif img.dim() == 3 and img.shape[0] == 1:
        arr = (img.clamp(0, 1) * 255.0).byte().squeeze(0).cpu().numpy()
        return Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unexpected tensor shape for image: {tuple(img.shape)}")



class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

def train_one_epoch(model, dl, optimizer, device, epoch, args, perceptual_loss_fn):
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
        
        # Re-compute mask_gt with patch_size=1 for U-Net (pixel-level mask)
        # mask_gt from dataloader is patch-level (e.g. 32x32), but U-Net outputs 256x256
        mask_gt = residual_soft_mask(
            x, 
            y, 
            patch_size=1, 
            temperature=32.0, # Default from dataset
            use_quantile=False, 
            quantile=0.9
        ).to(device)

        # l1 = nn.L1Loss()(out["y_hat"], y)
        char_loss = CharbonnierLoss()(out["y_hat"], y)
        mask_loss = nn.SmoothL1Loss()(m_hat, mask_gt)
        
        # Perceptual Loss
        p_loss = perceptual_loss_fn(out["y_hat"], y)
        
        loss = char_loss + args.lambda_mask * mask_loss + args.lambda_perceptual * p_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0:
            res_mag = out["residual_gated"].abs().mean().item()
            mask_gt_mean = mask_gt.mean().item()
            print(f"Epoch {epoch} [{i}/{len(dl)}] Loss: {loss.item():.4f} (Char: {char_loss.item():.4f}, Mask: {mask_loss.item():.4f}, Perc: {p_loss.item():.4f}, Res: {res_mag:.4f}, MaskGT: {mask_gt_mean:.4f})")

    avg_loss = total_loss / (i + 1)
    duration = time.time() - start_time
    print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}. Time: {duration:.2f}s")
    return avg_loss


def validate(model, dl, device, lpips_loss, epoch, save_dir):
    model.eval()
    psnr_acc = 0.0
    ssim_acc = 0.0
    lpips_acc = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lsb = batch["lsb"].to(device)
            msb = batch["msb"].to(device)
            
            out = model(x, lsb, msb)
            y_hat = out["y_hat"].clamp(0.0, 1.0)
            
            psnr_acc += calculate_psnr(y_hat, y)
            ssim_acc += calculate_ssim(y_hat, y)
            lpips_acc += lpips_loss(y_hat, y).item()
            
            # Visualization for specific files
            if "path_noisy" in batch:
                paths = batch["path_noisy"]
                # Debug: Print first path of first batch to verify
                if count == 0:
                    print(f"DEBUG: Validation paths sample: {paths[:2]}")
                
                # Define target substrings to match
                targets = [
                    "bathroom/000096",
                    "operating_room/000017",
                    "warehouse/000094",
                    "pantry/000115",
                    "office/000048"
                ]
                
                vis_dir = os.path.join(save_dir, "vis", f"epoch_{epoch}")
                
                for i, path in enumerate(paths):
                    if path is None: continue
                    # Check if any target substring is in the path
                    for t in targets:
                        if t in path:
                            os.makedirs(vis_dir, exist_ok=True)
                            # Create a filename safe version of the path or use the target name
                            safe_name = t.replace("/", "_")
                            
                            tensor_to_pil(x[i]).save(os.path.join(vis_dir, f"{safe_name}_noisy.png"))
                            tensor_to_pil(y[i]).save(os.path.join(vis_dir, f"{safe_name}_clean.png"))
                            tensor_to_pil(y_hat[i]).save(os.path.join(vis_dir, f"{safe_name}_denoised.png"))
                            tensor_to_pil(out["m_hat"][i]).save(os.path.join(vis_dir, f"{safe_name}_mask_pred.png"))
                            break # Found a match, move to next image in batch

            count += 1
            
    avg_psnr = psnr_acc / max(count, 1)
    avg_ssim = ssim_acc / max(count, 1)
    avg_lpips = lpips_acc / max(count, 1)
    
    log_msg = f"Epoch {epoch} Validation - PSNR: {avg_psnr:.3f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}"
    print(log_msg)
    
    with open(os.path.join(save_dir, "log.txt"), "a") as f:
        f.write(log_msg + "\n")
        
    return avg_psnr, avg_ssim, avg_lpips


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
    parser.add_argument("--lambda-perceptual", type=float, default=0.1, help="Weight for perceptual loss")
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

    # Validation Dataset
    val_ds = ExternalPairedBitPlaneDataset(
        module_path=args.external_module,
        root_dir=args.root,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        augment=False,
        split="val"  # Assuming 'val' split exists in external dataset
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Metrics & Losses
    lpips_loss_fn = LPIPS().to(args.device).eval() # For validation metrics
    perceptual_loss_fn = LPIPS().to(args.device).eval() # For training loss (freeze weights)

    # model = BitPlaneFormerV1(
    #     patch_size=args.patch_size,
    #     embed_dim=256,
    #     num_heads=8,
    #     msb_depth=6,
    #     dec_depth=6,
    #     mlp_ratio=4.0,
    #     dropout=0.0,
    # ).to(args.device)
    
    # Use U-Net
    model = BitPlaneUNet(n_channels=27, n_classes=3).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, dl, optimizer, args.device, epoch, args, perceptual_loss_fn)
        
        # Save checkpoint based on interval
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
            # Run validation
            validate(model, val_dl, args.device, lpips_loss_fn, epoch, args.save_dir)

if __name__ == "__main__":
    main()
