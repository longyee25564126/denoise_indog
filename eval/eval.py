import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from datasets.bitplane_utils import expand_bits
from models.bitplane_former_v1 import BitPlaneFormerV1

# Use the following command to evaluate model metrics: 
# python eval.py --root "/path to dataset" --checkpoint "/path to best.pth" --split "test" --external-module "/path to data_loader.py"


try:
    import lpips
except ImportError:
    print("Error: lpips not installed. pip install lpips")
    sys.exit(1)

# ==========================================
# Visualization Helpers
# ==========================================
def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        return Image.fromarray(arr.squeeze(2), mode="L")
    return Image.fromarray(arr, mode="RGB")

def save_comparison_image(noisy, denoised, gt, filename, save_dir):
    img_noisy = tensor_to_pil(noisy)
    img_denoised = tensor_to_pil(denoised)
    img_gt = tensor_to_pil(gt)
    
    w, h = img_gt.size
    header_h = 30
    gap = 10
    total_w = w * 3 + gap * 2
    total_h = h + header_h
    
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None

    items = [
        ("Noisy", img_noisy),
        ("Denoised", img_denoised),
        ("Ground Truth", img_gt)
    ]
    
    for i, (label, img) in enumerate(items):
        x_offset = i * (w + gap)
        draw.text((x_offset + 5, 5), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_offset, header_h))
        
    save_path = os.path.join(save_dir, filename)
    canvas.save(save_path)


# ==========================================
# Metric Calculations
# ==========================================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item() + eps)

def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, window_sigma: float = 1.5) -> float:
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * window_sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    window_1d = g
    window_2d = window_1d.transpose(1, 2) @ window_1d
    window = window_2d.expand(3, 1, window_size, window_size)

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

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

# ==========================================
# Data Loading
# ==========================================
def load_dataset(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("external_dataloader", args.external_module)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {args.external_module}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    try:
        base_dataset = module.PairedImageDataset(root_dir=args.root, split=args.split, transform=None)
    except TypeError:
        base_dataset = module.PairedImageDataset(root_dir=args.root, transform=None)

    ds = ExternalPairedBitPlaneDataset(
        base_dataset=base_dataset,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        crop_size=args.crop_size,
        augment=False,
        split=args.split,
        fit_to_patch=True,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits
    )
    return ds

def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics and save visualizations.")
    parser.add_argument("--root", type=str, required=True, help="dataset root")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=str, default="eval_results.csv")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-dir", type=str, default="eval_images")
    parser.add_argument("--max-images", type=int, default=-1)

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"\n[Info] Device: {device}")
    if device.type == 'cpu':
        print("[Warning] Using CPU, this might be slow.")

    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    
    dec_type = ckpt_args.get("dec_type", "fuse_encoder")
    # Use args.patch_stride if provided, else use checkpoint, else use patch_size
    if args.patch_stride is None:
        args.patch_stride = ckpt_args.get("patch_stride", args.patch_size)
    
    pad_size = ckpt_args.get("pad_size", 0)

    args.lsb_bits = expand_bits(args.lsb_bits)
    args.msb_bits = expand_bits(args.msb_bits)
    
    # Note: args.crop_size is passed to load_dataset via args

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        pad_size=pad_size,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
        dec_type=dec_type
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 2. LPIPS
    print("Loading LPIPS model (VGG)...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.eval()

    # 3. Load Dataset
    ds = load_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset loaded. Size: {len(ds)}")

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[Info] Saving images to: {os.path.abspath(args.save_dir)}")

    # 4. Evaluation Loop
    results_list = []
    saved_img_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lsb = batch["lsb"].to(device)
            msb = batch["msb"].to(device)

            filenames = batch.get("path_noisy", [None] * x.size(0))
            labels = batch.get("label", [None] * x.size(0))
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().tolist()

            out = model({"x": x, "lsb": lsb, "msb": msb})
            y_hat = out["y_hat"].clamp(0.0, 1.0)

            # LPIPS Input [-1, 1]
            p_gt = (y * 2 - 1).clamp(-1, 1)
            p_noisy = (x * 2 - 1).clamp(-1, 1)
            p_denoised = (y_hat * 2 - 1).clamp(-1, 1)
            
            lpips_noisy_batch = loss_fn_lpips(p_noisy, p_gt).view(-1).cpu().tolist()
            lpips_denoised_batch = loss_fn_lpips(p_denoised, p_gt).view(-1).cpu().tolist()

            batch_sz = x.size(0)
            for b in range(batch_sz):
                psnr_noisy = compute_psnr(x[b], y[b])
                ssim_noisy = compute_ssim(x[b], y[b])
                psnr_denoised = compute_psnr(y_hat[b], y[b])
                ssim_denoised = compute_ssim(y_hat[b], y[b])
                
                filename = filenames[b]
                if filename is None:
                    filename = f"sample_{i * args.batch_size + b}.png"
                else:
                    filename = os.path.basename(str(filename))

                record = {
                    "Filename": filename,
                    "Label": labels[b],
                    "PSNR_Noisy": psnr_noisy,
                    "SSIM_Noisy": ssim_noisy,
                    "LPIPS_Noisy": lpips_noisy_batch[b],
                    "PSNR_Denoised": psnr_denoised,
                    "SSIM_Denoised": ssim_denoised,
                    "LPIPS_Denoised": lpips_denoised_batch[b]
                }
                results_list.append(record)

                if args.save_images:
                    if args.max_images < 0 or saved_img_count < args.max_images:
                        save_name = filename
                        if not save_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            save_name += ".png"
                        
                        save_comparison_image(
                            noisy=x[b],
                            denoised=y_hat[b],
                            gt=y[b],
                            filename=save_name,
                            save_dir=args.save_dir
                        )
                        saved_img_count += 1

    # 5. Statistics & Save CSV
    df = pd.DataFrame(results_list)
    cols = ["Filename", "Label", 
            "PSNR_Noisy", "SSIM_Noisy", "LPIPS_Noisy", 
            "PSNR_Denoised", "SSIM_Denoised", "LPIPS_Denoised"]
    df = df[cols]
    
    avg_row = df.mean(numeric_only=True)
    
    print("\n" + "="*50)
    print(f"Evaluation Results (over {len(df)} images)")
    print("="*50)
    print(f"{'Metric':<10} | {'Input (Noisy)':<15} | {'Output (Denoised)':<15}")
    print("-" * 45)
    print(f"{'PSNR':<10} | {avg_row['PSNR_Noisy']:<15.4f} | {avg_row['PSNR_Denoised']:<15.4f}")
    print(f"{'SSIM':<10} | {avg_row['SSIM_Noisy']:<15.4f} | {avg_row['SSIM_Denoised']:<15.4f}")
    print(f"{'LPIPS':<10} | {avg_row['LPIPS_Noisy']:<15.4f} | {avg_row['LPIPS_Denoised']:<15.4f}")
    print("="*50)

    # Save CSV Only
    df.to_csv(args.output_csv, index=False)
    print(f"[Success] Results saved to CSV: {os.path.abspath(args.output_csv)}")

    if args.save_images:
        print(f"[Success] Saved {saved_img_count} images to: {os.path.abspath(args.save_dir)}")

if __name__ == "__main__":
    main()