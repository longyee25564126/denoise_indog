import argparse
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from datasets.bitplane_utils import to_uint8, expand_bits
from models.bitplane_former_v1 import BitPlaneFormerV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask ablation on validation set: pred vs all-ones vs GT.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root.")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--fit-to-patch", action="store_true")
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-T", type=float, default=48.0)
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples for quick eval.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> Tuple[ExternalPairedBitPlaneDataset, dict]:
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
        augment=False,
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
    return ds, {"size": len(ds)}


def compute_mask_gt(x: torch.Tensor, y: torch.Tensor, patch_size: int, temperature: float) -> torch.Tensor:
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


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    args.lsb_bits = expand_bits(args.lsb_bits)
    args.msb_bits = expand_bits(args.msb_bits)

    ds, info = load_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint}; dataset size {info['size']}")

    total = 0
    psnr_pred = 0.0
    psnr_one = 0.0
    psnr_gt = 0.0

    with torch.no_grad():
        for batch in dl:
            if args.limit > 0 and total >= args.limit:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lsb = batch["lsb"].to(device)
            msb = batch["msb"].to(device)

            if args.limit > 0 and total + x.size(0) > args.limit:
                keep = args.limit - total
                x, y, lsb, msb = x[:keep], y[:keep], lsb[:keep], msb[:keep]

            mask_gt = compute_mask_gt(x, y, args.patch_size, args.mask_T)
            ones_mask = torch.ones_like(mask_gt)

            out_pred = model({"x": x, "lsb": lsb, "msb": msb})
            out_one = model({"x": x, "lsb": lsb, "msb": msb}, mask_override=ones_mask)
            out_gt = model({"x": x, "lsb": lsb, "msb": msb}, mask_override=mask_gt)

            y_hat_pred = out_pred["y_hat"].clamp(0, 1)
            y_hat_one = out_one["y_hat"].clamp(0, 1)
            y_hat_gt = out_gt["y_hat"].clamp(0, 1)

            for b in range(x.size(0)):
                psnr_pred += psnr(y_hat_pred[b], y[b])
                psnr_one += psnr(y_hat_one[b], y[b])
                psnr_gt += psnr(y_hat_gt[b], y[b])
            total += x.size(0)

    psnr_pred /= max(total, 1)
    psnr_one /= max(total, 1)
    psnr_gt /= max(total, 1)

    print(f"Samples evaluated: {total}")
    print(f"PSNR pred_mask : {psnr_pred:.3f}")
    print(f"PSNR mask=ones : {psnr_one:.3f}")
    print(f"PSNR mask=gt   : {psnr_gt:.3f}")


if __name__ == "__main__":
    main()
