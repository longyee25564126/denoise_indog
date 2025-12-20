import argparse
import importlib.util
import json
import os
import sys
from typing import Any, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset  # noqa: E402
from datasets.bitplane_utils import to_uint8  # noqa: E402


def _load_paired_dataset(module_path: str, root_dir: str, split: str):
    spec = importlib.util.spec_from_file_location("external_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PairedImageDataset"):
        raise AttributeError(f"Module {module_path} has no PairedImageDataset")
    PairedImageDataset = getattr(module, "PairedImageDataset")
    return PairedImageDataset(root_dir=root_dir, split=split, transform=transforms.ToTensor())


def format_header() -> str:
    return "     " + " ".join(f"{('b'+str(k)):>8}" for k in range(8))


def format_row(name: str, values: torch.Tensor) -> str:
    vals = " ".join(f"{v.item()*100:7.4f}%" for v in values)
    return f"{name:>4}: {vals}"


def find_threshold_hits(avg_rates: torch.Tensor, thresholds=(0.02, 0.05)):
    hits = {}
    for thr in thresholds:
        hit = None
        for k, v in enumerate(avg_rates):
            if v.item() < thr:
                hit = k
                break
        hits[thr] = hit
    return hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bit flip rate (noisy vs clean) over a dataset.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root for PairedImageDataset.")
    parser.add_argument("--split", type=str, required=True, help="Split name (e.g., train/val/test).")
    parser.add_argument(
        "--external-module",
        type=str,
        default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py",
        help="Path to data_loader.py defining PairedImageDataset.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers.")
    parser.add_argument("--patch-size", type=int, required=True, help="Patch size for divisibility checks in wrapper.")
    parser.add_argument("--crop-size", type=int, default=None, help="Optional crop size (must divide patch size).")
    parser.add_argument("--augment", action="store_true", help="Enable flips/rotations; usually keep off for stats.")
    parser.add_argument("--fit-to-patch", action="store_true", help="Center-crop to nearest patch-aligned size if needed.")
    parser.add_argument("--limit", type=int, default=-1, help="If >0, only process the first N samples.")
    parser.add_argument("--device", type=str, default=None, help='Device to use (default: "cuda" if available else "cpu").')
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save stats as JSON.")
    return parser.parse_args()


def _collate_list(batch: List[Any]) -> List[Any]:
    """
    Custom collate_fn that keeps a list of samples (no stacking), so variable image sizes won't break.
    """
    return batch


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    base_ds = _load_paired_dataset(args.external_module, args.root, args.split)
    ds = ExternalPairedBitPlaneDataset(
        base_dataset=base_ds,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        augment=args.augment,
        return_mask_flat=False,
        split=args.split,
        fit_to_patch=args.fit_to_patch,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_list,
    )

    mismatch = torch.zeros(3, 8, dtype=torch.int64, device=device)
    total = torch.zeros(3, 8, dtype=torch.int64, device=device)

    processed = 0
    limit = args.limit if args.limit is not None else -1
    max_samples = len(ds) if limit < 0 else min(limit, len(ds))

    for batch in dl:
        if limit > 0 and processed >= limit:
            break

        for sample in batch:
            if limit > 0 and processed >= limit:
                break
            x = sample["x"].to(device)
            y = sample["y"].to(device)

            x_u8 = to_uint8(x)
            y_u8 = to_uint8(y)

            for c in range(3):
                for k in range(8):
                    xb = (x_u8[c] >> k) & 1
                    yb = (y_u8[c] >> k) & 1
                    diff = torch.count_nonzero(xb != yb)
                    mismatch[c, k] += diff
                    total[c, k] += xb.numel()

            processed += 1

    rates = torch.zeros_like(mismatch, dtype=torch.float32)
    nonzero = total > 0
    rates[nonzero] = mismatch[nonzero].float() / total[nonzero].float()
    rates_cpu = rates.cpu()
    avg_rates = rates_cpu.mean(dim=0)

    print(f"Dataset size: {len(ds)} samples; processed: {processed}")
    print("Flip rates per channel (percent):")
    print(format_header())
    print(format_row("R", rates_cpu[0]))
    print(format_row("G", rates_cpu[1]))
    print(format_row("B", rates_cpu[2]))
    print(format_row("AVG", avg_rates))

    hits = find_threshold_hits(avg_rates, thresholds=(0.02, 0.05))
    msg_parts = []
    for thr in (0.02, 0.05):
        hit = hits[thr]
        if hit is not None:
            msg_parts.append(f"<{thr}: bit {hit} (flip {avg_rates[hit].item():.4f})")
        else:
            msg_parts.append(f"<{thr}: none")
    print("Suggested MSB clean start (avg flip rate): " + ", ".join(msg_parts))

    if args.save_json:
        out = {
            "rates_rgb": rates_cpu.tolist(),
            "rates_avg": avg_rates.tolist(),
            "dataset_size": len(ds),
            "processed": processed,
            "args": vars(args),
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved stats to {args.save_json}")


if __name__ == "__main__":
    main()
