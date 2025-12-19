import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from datasets.bitplane_dataset import BitPlanePairDataset  # noqa: E402
from datasets.external_adapter import ExternalPairedBitPlaneDataset  # noqa: E402
from models.bitplane_former_v1 import BitPlaneFormerV1  # noqa: E402


def _make_dummy_dataset(root: str, num_images: int = 4, size: int = 256) -> None:
    noisy_dir = os.path.join(root, "noisy")
    clean_dir = os.path.join(root, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    for i in range(num_images):
        noisy = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        clean = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(noisy).save(os.path.join(noisy_dir, f"img_{i}.png"))
        Image.fromarray(clean).save(os.path.join(clean_dir, f"img_{i}.png"))


def build_dataset(args):
    using_tmp = args.root is None
    tmp_root = None
    if args.use_external:
        if args.root is None:
            raise ValueError("--use-external requires --root")
        ds = ExternalPairedBitPlaneDataset(
            module_path=args.external_module,
            root_dir=args.root,
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            augment=args.augment,
            return_mask_flat=False,
            split=args.split,
        )
    else:
        data_root = args.root
        if using_tmp:
            tmp_root = tempfile.mkdtemp(prefix="bitplane_ds_")
            _make_dummy_dataset(tmp_root, num_images=args.batch_size + 2, size=args.crop_size)
            data_root = tmp_root
        ds = BitPlanePairDataset(
            root=data_root,
            pairs_file=args.pairs_file,
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            augment=args.augment,
            strict_pairing=True,
            return_mask_flat=False,
        )
    return ds, tmp_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward test for BitPlaneFormerV1.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root. If absent, uses dummy data.")
    parser.add_argument("--pairs-file", type=str, default=None, help="Optional pairs.txt for internal loader.")
    parser.add_argument("--use-external", action="store_true", help="Use ExternalPairedBitPlaneDataset.")
    parser.add_argument(
        "--external-module",
        type=str,
        default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py",
        help="Path to external data_loader.py defining PairedImageDataset.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when using external loader.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ds, tmp_root = build_dataset(args)
    try:
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        batch = next(iter(dl))

        device = torch.device(args.device)
        model = BitPlaneFormerV1(
            patch_size=args.patch_size,
            embed_dim=256,
            num_heads=8,
            msb_depth=6,
            dec_depth=6,
            mlp_ratio=4.0,
            dropout=0.0,
        ).to(device)

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lsb = batch["lsb"].to(device)
        msb = batch["msb"].to(device)
        mask_gt = batch["mask_gt"].to(device)

        out = model(x, lsb, msb)

        m_logits = out["m_logits"]
        # Ensure mask_gt shape matches m_logits
        if mask_gt.ndim != m_logits.ndim:
            B, _, h, w = m_logits.shape
            mask_gt = mask_gt.view(B, 1, h, w)

        l1 = nn.L1Loss()(out["y_hat"], y)
        bce = nn.BCEWithLogitsLoss()(m_logits, mask_gt)
        loss = l1 + 0.1 * bce
        loss.backward()

        print(f"x: {tuple(x.shape)}, y_hat: {tuple(out['y_hat'].shape)}")
        print(f"residual_gated: {tuple(out['residual_gated'].shape)}")
        print(f"m_logits: {tuple(m_logits.shape)}, mask_gt: {tuple(mask_gt.shape)}")
        print(f"L1: {l1.item():.4f}, BCE: {bce.item():.4f}, total: {loss.item():.4f}")
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
