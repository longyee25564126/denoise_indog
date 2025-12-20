import argparse
import os
import sys
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from datasets.bitplane_utils import to_uint8, expand_bits
from models.bitplane_former_v1 import BitPlaneFormerV1


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


def compute_mask_gt(x: torch.Tensor, y: torch.Tensor, patch_size: int, temperature: float) -> torch.Tensor:
    x_u8 = to_uint8(x)
    y_u8 = to_uint8(y)
    err = (x_u8.to(torch.float32) - y_u8.to(torch.float32)).abs().mean(dim=1, keepdim=True)
    mask_pix = torch.clamp(err / temperature, 0.0, 1.0)
    mask_gt = F.avg_pool2d(mask_pix, kernel_size=patch_size, stride=patch_size)
    return mask_gt


def load_dataset(args: argparse.Namespace, train: bool = False) -> Tuple[ExternalPairedBitPlaneDataset, dict]:
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

    ds = ExternalPairedBitPlaneDataset(
        base_dataset=base_dataset,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        augment=train and args.augment,
        return_mask_flat=False,
        split=args.split,
        fit_to_patch=args.fit_to_patch,
        use_residual_mask=True,
        mask_temperature=args.mask_T,
        mask_use_quantile=False,
        mask_quantile=0.9,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
    )
    info = {"size": len(ds)}
    return ds, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BitPlaneFormerV1 on a single sample and visualize outputs.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root.")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py", help="Path to data_loader.py.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to sample from.")
    parser.add_argument("--index", type=int, default=-1, help="Sample index to visualize; -1 for random.")
    parser.add_argument("--name", type=str, default=None, help="Substring to pick sample by filename (noisy or clean).")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--augment", action="store_true", help="Enable augment for loading (usually off for eval).")
    parser.add_argument("--fit-to-patch", action="store_true", help="Center-crop to nearest patch-aligned size.")
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--mask-T", type=float, default=48.0, help="Temperature for residual mask.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help='Device (default: "cuda" if available else "cpu").')
    parser.add_argument("--save-dir", type=str, default="outputs/infer_vis", help="Directory to save visualizations.")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    args.lsb_bits = expand_bits(args.lsb_bits)
    args.msb_bits = expand_bits(args.msb_bits)

    ds, info = load_dataset(args, train=False)
    idx = args.index
    if args.name is not None:
        matches = []
        target = args.name.lower()
        for i, (noisy_path, clean_path, _) in enumerate(ds.base_dataset.samples):
            if target in noisy_path.lower() or target in clean_path.lower():
                matches.append(i)
        if not matches:
            raise ValueError(f"No sample filename contains '{args.name}' in split {args.split}")
        idx = matches[0]
        print(f"Found name match '{args.name}' at index {idx} (first match of {len(matches)})")
    elif idx < 0:
        idx = random.randrange(len(ds))
    else:
        idx = max(0, min(idx, len(ds) - 1))
    sample = ds[idx]

    x = sample["x"].unsqueeze(0).to(device)  # B=1
    y = sample["y"].unsqueeze(0).to(device)
    lsb = sample["lsb"].unsqueeze(0).to(device)
    msb = sample["msb"].unsqueeze(0).to(device)

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.eval()

    with torch.no_grad():
        out = model({"x": x, "lsb": lsb, "msb": msb})
        mask_gt = compute_mask_gt(x, y, args.patch_size, args.mask_T)

    y_hat = out["y_hat"].clamp(0.0, 1.0).squeeze(0)
    x_vis = x.squeeze(0)
    y_vis = y.squeeze(0)
    m_pred = out["m_hat"].detach().cpu().squeeze(0)
    m_gt = mask_gt.cpu().squeeze(0)

    os.makedirs(args.save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(sample.get("path_noisy") or f"idx_{idx}"))[0]
    paths = {
        "noisy": os.path.join(args.save_dir, f"{stem}_noisy.png"),
        "clean": os.path.join(args.save_dir, f"{stem}_clean.png"),
        "denoised": os.path.join(args.save_dir, f"{stem}_denoised.png"),
        "mask_pred": os.path.join(args.save_dir, f"{stem}_mask_pred.png"),
        "mask_gt": os.path.join(args.save_dir, f"{stem}_mask_gt.png"),
    }

    tensor_to_pil(x_vis).save(paths["noisy"])
    tensor_to_pil(y_vis).save(paths["clean"])
    tensor_to_pil(y_hat).save(paths["denoised"])
    tensor_to_pil(m_pred).save(paths["mask_pred"])
    tensor_to_pil(m_gt).save(paths["mask_gt"])

    print(f"Dataset size {info['size']}, visualized index {idx}")
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(
        f"mask_pred mean/std: {m_pred.mean().item():.4f}/{m_pred.std().item():.4f} | "
        f"mask_gt mean/std: {m_gt.mean().item():.4f}/{m_gt.std().item():.4f}"
    )


if __name__ == "__main__":
    main()
