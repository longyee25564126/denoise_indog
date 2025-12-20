import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Add root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from youming_models.external_adapter import ExternalPairedBitPlaneDataset
from youming_models.unet_model import BitPlaneUNet
from youming_models.train import tensor_to_pil
from datasets.bitplane_utils import make_lsb_msb, to_uint8
import numpy as np
from torch.utils.data import Dataset

class SimpleFolderDataset(Dataset):
    def __init__(self, root_dir, clean_dir=None):
        self.root_dir = root_dir
        self.clean_dir = clean_dir
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Try to load clean image if clean_dir is provided
        y = x.clone() # Default to noisy if clean not found
        has_clean = False
        
        if self.clean_dir:
            # Logic to find clean image
            # Assumption: noisy path is .../class/name_noisy.ext
            # Clean path is .../class/name.ext or name_clean.ext
            
            rel_path = os.path.relpath(path, self.root_dir)
            parent, filename = os.path.split(rel_path)
            name_no_ext, ext = os.path.splitext(filename)
            
            # Try removing "_noisy"
            clean_name_candidates = [
                filename.replace("_noisy", ""),
                filename.replace("_noisy", "_clean"),
                filename
            ]
            
            for cand in clean_name_candidates:
                clean_path = os.path.join(self.clean_dir, parent, cand)
                if os.path.exists(clean_path):
                    clean_img = Image.open(clean_path).convert('RGB')
                    y = torch.from_numpy(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
                    has_clean = True
                    break
        
        # Ensure dimensions are divisible by 8
        _, h, w = x.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        x = x[:, :new_h, :new_w]
        y = y[:, :new_h, :new_w]
        
        x_u8 = to_uint8(x)
        lsb, msb = make_lsb_msb(x_u8)
        
        return {
            "x": x,
            "lsb": lsb,
            "msb": msb,
            "path_noisy": path,
            "y": y,
            "has_clean": has_clean
        }

def main():
    parser = argparse.ArgumentParser(description="Inference for BitPlaneUNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--root", type=str, default="/home/youming/dataset_and_data_loader/dataset final version", help="Dataset root")
    parser.add_argument(
        "--external-module",
        type=str,
        default="/home/youming/dataset_and_data_loader/data_loader.py",
        help="Path to external data_loader.py",
    )
    parser.add_argument("--output-dir", type=str, default="youming_models/output/inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use (train/val/test)")
    parser.add_argument("--image-dir", type=str, default=None, help="Directly load images from this directory (bypassing data_loader.py)")
    parser.add_argument("--clean-dir", type=str, default=None, help="Directory containing clean images (optional)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = BitPlaneUNet(n_channels=27, n_classes=3).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Validation Dataset
    if args.image_dir:
        print(f"Loading images directly from {args.image_dir}...")
        if args.clean_dir:
            print(f"Loading clean images from {args.clean_dir}...")
        val_ds = SimpleFolderDataset(args.image_dir, args.clean_dir)
    else:
        print(f"Loading dataset from {args.root}...")
        val_ds = ExternalPairedBitPlaneDataset(
            module_path=args.external_module,
            root_dir=args.root,
            patch_size=8,
            crop_size=256,
            augment=False,
            split=args.split
        )
    
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    targets = [
        "bathroom/000096",
        "operating_room/000017",
        "warehouse/000094",
        "pantry/000115",
        "office/000048"
    ]
    
    print("Starting inference...")
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            path = batch["path_noisy"][0] if "path_noisy" in batch else None
            
            if i < 5:
                print(f"DEBUG: Checking image {i}, path: {path}")

            # Check if this image is in our target list
            is_target = False
            if path:
                for t in targets:
                    if t in path:
                        is_target = True
                        break
            
            if not is_target:
                continue
                
            print(f"Processing {path}...")
            
            x = batch["x"].to(args.device)
            lsb = batch["lsb"].to(args.device)
            msb = batch["msb"].to(args.device)
            y = batch["y"].to(args.device)
            
            out = model(x, lsb, msb)
            y_hat = out["y_hat"].clamp(0.0, 1.0)
            m_hat = out["m_hat"]
            
            # Shorten filename: parent_folder + "_" + filename
            # e.g. .../bathroom/000096_noisy.jpg -> bathroom_000096_noisy.jpg
            parent = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            short_name = f"{parent}_{name}"
            
            tensor_to_pil(x[0]).save(os.path.join(args.output_dir, f"{short_name}_input.png"))
            # User requested "_clean" suffix for the result
            tensor_to_pil(y_hat[0]).save(os.path.join(args.output_dir, f"{short_name}_clean.png"))
            tensor_to_pil(m_hat[0]).save(os.path.join(args.output_dir, f"{short_name}_mask.png"))
            
            # Save GT if available
            if "has_clean" in batch and batch["has_clean"][0]:
                 tensor_to_pil(y[0]).save(os.path.join(args.output_dir, f"{short_name}_gt.png"))
            
            count += 1
            if count >= len(targets):
                break
                
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
