import argparse
import os
import sys
import random
import torch
import torch.nn.functional as F
from PIL import Image

# Add root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from youming_models.external_adapter import ExternalPairedBitPlaneDataset
from youming_models.bitplane_former_v1 import BitPlaneFormerV1
from datasets.bitplane_utils import to_uint8

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

def main():
    parser = argparse.ArgumentParser(description="Visualize Youming Model Results")
    parser.add_argument("--root", type=str, default="/home/youming/dataset_and_data_loader/dataset final version", help="Dataset root")
    parser.add_argument("--external-module", type=str, default="/home/youming/dataset_and_data_loader/data_loader.py", help="Path to data_loader.py")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save-dir", type=str, default="youming_models/output/vis", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Dataset
    print(f"Loading dataset from {args.root}...")
    ds = ExternalPairedBitPlaneDataset(
        module_path=args.external_module,
        root_dir=args.root,
        patch_size=args.patch_size,
        crop_size=256, # Use fixed crop size for visualization
        augment=False,
        split="val" # Try to use validation set
    )
    print(f"Dataset size: {len(ds)}")

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        embed_dim=256,
        num_heads=8,
        msb_depth=6,
        dec_depth=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ).to(device)
    
    # Load state dict
    # Note: The training script saves the whole state_dict directly, not wrapped in "model" key
    # But let's check if it's wrapped or not. Based on train.py: torch.save(model.state_dict(), ckpt_path)
    # So it is NOT wrapped.
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    indices = random.sample(range(len(ds)), min(args.num_samples, len(ds)))
    
    print(f"Visualizing {len(indices)} samples...")
    
    for i, idx in enumerate(indices):
        sample = ds[idx]
        
        x = sample["x"].unsqueeze(0).to(device)
        y = sample["y"].unsqueeze(0).to(device)
        lsb = sample["lsb"].unsqueeze(0).to(device)
        msb = sample["msb"].unsqueeze(0).to(device)
        mask_gt = sample["mask_gt"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(x, lsb, msb)
            
        y_hat = out["y_hat"].clamp(0.0, 1.0)
        m_hat = out["m_hat"]
        residual = out["residual_gated"]

        print(f"Sample {idx}:")
        l1_xy = (x - y).abs().mean().item()
        print(f"  L1(x, y): {l1_xy:.4f}")
        print(f"  Mask Pred Mean: {m_hat.mean().item():.4f}, Std: {m_hat.std().item():.4f}")
        print(f"  Mask GT Mean: {mask_gt.mean().item():.4f}, Std: {mask_gt.std().item():.4f}")
        print(f"  Residual Mean: {residual.abs().mean().item():.4f}, Max: {residual.abs().max().item():.4f}")
        
        # Save images
        stem = f"sample_{idx}"
        
        tensor_to_pil(x.squeeze(0)).save(os.path.join(args.save_dir, f"{stem}_noisy.png"))
        tensor_to_pil(y.squeeze(0)).save(os.path.join(args.save_dir, f"{stem}_clean.png"))
        tensor_to_pil(y_hat.squeeze(0)).save(os.path.join(args.save_dir, f"{stem}_denoised.png"))
        tensor_to_pil(m_hat.squeeze(0)).save(os.path.join(args.save_dir, f"{stem}_mask_pred.png"))
        tensor_to_pil(mask_gt.squeeze(0)).save(os.path.join(args.save_dir, f"{stem}_mask_gt.png"))
        
        print(f"Saved results for sample {idx} to {args.save_dir}")

if __name__ == "__main__":
    main()
