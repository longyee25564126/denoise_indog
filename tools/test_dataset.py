import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from datasets.bitplane_dataset import BitPlanePairDataset
from datasets.external_adapter import ExternalPairedBitPlaneDataset


def _make_dummy_dataset(root: str, num_images: int = 3, size: int = 256) -> None:
    noisy_dir = os.path.join(root, "noisy")
    clean_dir = os.path.join(root, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    for i in range(num_images):
        noisy = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        clean = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(noisy).save(os.path.join(noisy_dir, f"img_{i}.png"))
        Image.fromarray(clean).save(os.path.join(clean_dir, f"img_{i}.png"))


def describe_sample(sample: dict, patch_size: int, patch_stride: int) -> None:
    x, y = sample["x"], sample["y"]
    lsb, msb = sample["lsb"], sample["msb"]
    mask_gt = sample["mask_gt"]
    H, W = x.shape[1], x.shape[2]
    h = (H - patch_size) // patch_stride + 1
    w = (W - patch_size) // patch_stride + 1

    print(f"x shape {tuple(x.shape)}, dtype {x.dtype}, min/max {x.min().item():.3f}/{x.max().item():.3f}")
    print(f"y shape {tuple(y.shape)}, dtype {y.dtype}, min/max {y.min().item():.3f}/{y.max().item():.3f}")
    print(f"lsb shape {tuple(lsb.shape)}, dtype {lsb.dtype}, unique {torch.unique(lsb)}")
    print(f"msb shape {tuple(msb.shape)}, dtype {msb.dtype}, unique {torch.unique(msb)}")
    if mask_gt.ndim == 3:
        assert mask_gt.shape == (1, h, w)
    else:
        assert mask_gt.shape[0] == h * w
    print(f"mask_gt shape {tuple(mask_gt.shape)}, min/max {mask_gt.min().item():.3f}/{mask_gt.max().item():.3f}")
    assert H >= patch_size and W >= patch_size
    assert (H - patch_size) % patch_stride == 0 and (W - patch_size) % patch_stride == 0
    print(f"Passed alignment check P={patch_size}, S={patch_stride}")
    print(f"paths noisy={sample['path_noisy']}, clean={sample['path_clean']}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for BitPlanePairDataset.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (paired noisy/clean) or external dataset root.")
    parser.add_argument("--pairs-file", type=str, default=None, help="Optional pairs.txt when using --root with internal loader.")
    parser.add_argument("--use-external", action="store_true", help="Use external PairedImageDataset (noisy, clean, label).")
    parser.add_argument(
        "--external-module",
        type=str,
        default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py",
        help="Path to external data_loader.py that defines PairedImageDataset.",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--augment", action="store_true", help="Enable flip/rot90 augment.")
    parser.add_argument("--return-mask-flat", action="store_true")
    args = parser.parse_args()

    using_tmp = args.root is None
    tmp_root = None

    try:
        if args.patch_stride is None:
            args.patch_stride = args.patch_size
        if args.use_external:
            if args.root is None:
                raise ValueError("--use-external requires --root pointing to the external dataset root")
            ds = ExternalPairedBitPlaneDataset(
                module_path=args.external_module,
                root_dir=args.root,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                crop_size=args.crop_size,
                augment=args.augment,
                return_mask_flat=args.return_mask_flat,
            )
        else:
            data_root = args.root
            if using_tmp:
                tmp_root = tempfile.mkdtemp(prefix="bitplane_ds_")
                _make_dummy_dataset(tmp_root, num_images=args.num_samples, size=args.crop_size)
                data_root = tmp_root

            ds = BitPlanePairDataset(
                root=data_root,
                pairs_file=args.pairs_file,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                crop_size=args.crop_size,
                augment=args.augment,
                strict_pairing=True,
                return_mask_flat=args.return_mask_flat,
            )
        print(f"Dataset size: {len(ds)}; showing {min(args.num_samples, len(ds))} samples")
        for idx in range(min(args.num_samples, len(ds))):
            sample = ds[idx]
            describe_sample(sample, ds.patch_size, ds.patch_stride)
    finally:
        if using_tmp and tmp_root is not None:
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
