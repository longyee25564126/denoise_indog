import argparse
import os
import random
import sys
from typing import Tuple

import torch
from torchvision import transforms

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.external_adapter import ExternalPairedBitPlaneDataset
from datasets.bitplane_utils import expand_bits, make_lsb_msb, residual_soft_mask, to_uint8
from models.bitplane_former_v1 import BitPlaneFormerV1


def _load_external_dataset(args: argparse.Namespace) -> Tuple[ExternalPairedBitPlaneDataset, dict]:
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
        base_dataset = PairedImageDataset(root_dir=args.root, split=args.split, transform=transforms.ToTensor())
    except TypeError:
        base_dataset = PairedImageDataset(root_dir=args.root, transform=transforms.ToTensor())

    ds = ExternalPairedBitPlaneDataset(
        base_dataset=base_dataset,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        crop_size=args.crop_size,
        augment=args.augment,
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


def _pick_index(ds: ExternalPairedBitPlaneDataset, args: argparse.Namespace) -> int:
    if args.name is not None:
        matches = []
        target = args.name.lower()
        samples = getattr(ds.base_dataset, "samples", None)
        if samples is None:
            raise ValueError("Dataset does not expose samples; cannot match by name.")
        for i, (noisy_path, clean_path, _) in enumerate(samples):
            if target in noisy_path.lower() or target in clean_path.lower():
                matches.append(i)
        if not matches:
            raise ValueError(f"No sample filename contains '{args.name}' in split {args.split}")
        idx = matches[0]
        print(f"Found name match '{args.name}' at index {idx} (first match of {len(matches)})")
        return idx

    if args.index >= 0:
        return max(0, min(args.index, len(ds) - 1))

    return random.randrange(len(ds))


def _describe(name: str, t: torch.Tensor | None) -> None:
    if t is None:
        print(f"{name}: None")
        return
    stats = ""
    if t.is_floating_point():
        stats = f"min {t.min().item():.4f} max {t.max().item():.4f} mean {t.mean().item():.4f} std {t.std().item():.4f}"
    else:
        stats = f"min {t.min().item()} max {t.max().item()}"
    print(f"{name}: shape {tuple(t.shape)} dtype {t.dtype} {stats}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace the bit-plane pipeline on a single sample.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (required unless --dummy).")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--index", type=int, default=-1, help="Sample index; -1 for random.")
    parser.add_argument("--name", type=str, default=None, help="Substring to pick sample by filename.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--pad-size", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fit-to-patch", action="store_true")
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--mask-T", type=float, default=48.0)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to load.")
    parser.add_argument("--dec-type", type=str, default=None, choices=["fuse_encoder", "decoder_q_msb", "std_encdec_msb"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dummy", action="store_true", help="Use a small synthetic tensor instead of dataset.")
    parser.add_argument("--dummy-h", type=int, default=64, help="Dummy image height.")
    parser.add_argument("--dummy-w", type=int, default=64, help="Dummy image width.")
    parser.add_argument("--dummy-noise-std", type=float, default=0.05, help="Std of Gaussian noise for dummy input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_stride is None:
        args.patch_stride = args.patch_size
    args.lsb_bits = expand_bits(args.lsb_bits)
    args.msb_bits = expand_bits(args.msb_bits)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if args.dummy:
        H, W = args.dummy_h, args.dummy_w
        if H < args.patch_size or W < args.patch_size:
            raise ValueError("Dummy size must be >= patch_size.")
        if (H - args.patch_size) % args.patch_stride != 0 or (W - args.patch_size) % args.patch_stride != 0:
            raise ValueError("Dummy size must align with patch_size/patch_stride.")
        y = torch.rand(1, 3, H, W, device=device)
        noise = torch.randn_like(y) * args.dummy_noise_std
        x = (y + noise).clamp(0.0, 1.0)
        lsb, msb = make_lsb_msb(to_uint8(x.squeeze(0)), args.lsb_bits, args.msb_bits)
        lsb = lsb.unsqueeze(0).to(device)
        msb = msb.unsqueeze(0).to(device)
        mask_gt = residual_soft_mask(x, y, args.patch_size, args.patch_stride, args.mask_T)
        print(f"Using dummy tensor: H={H}, W={W}, noise_std={args.dummy_noise_std}")
    else:
        if args.root is None:
            raise ValueError("Please provide --root or use --dummy.")
        ds, info = _load_external_dataset(args)
        idx = _pick_index(ds, args)
        sample = ds[idx]
        print(f"Dataset size {info['size']}, index {idx}")
        print(f"path_noisy: {sample.get('path_noisy')}")
        print(f"path_clean: {sample.get('path_clean')}")

        x = sample["x"].unsqueeze(0).to(device)
        y = sample["y"].unsqueeze(0).to(device)
        lsb = sample["lsb"].unsqueeze(0).to(device)
        msb = sample["msb"].unsqueeze(0).to(device)
        mask_gt = sample.get("mask_gt")
        if mask_gt is not None:
            if mask_gt.ndim == 3:
                mask_gt = mask_gt.unsqueeze(0)
            mask_gt = mask_gt.to(device)

    print("\n=== Dataset Outputs ===")
    _describe("x (noisy)", x)
    _describe("y (clean)", y)
    _describe("lsb", lsb)
    _describe("msb", msb)
    _describe("mask_gt", mask_gt)

    model = BitPlaneFormerV1(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        pad_size=args.pad_size,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
        dec_type=args.dec_type or "fuse_encoder",
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "args" in ckpt and args.dec_type is None:
            ckpt_dec = ckpt.get("args", {}).get("dec_type")
            if ckpt_dec and ckpt_dec != model.dec_type:
                model = BitPlaneFormerV1(
                    patch_size=args.patch_size,
                    patch_stride=args.patch_stride,
                    lsb_bits=args.lsb_bits,
                    msb_bits=args.msb_bits,
                    dec_type=ckpt_dec,
                ).to(device)
                print(f"Rebuilt model with dec_type from checkpoint: {ckpt_dec}")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()
    with torch.no_grad():
        T_msb0, grid_shape = model.msb_tokenizer(msb)
        T_msb = model.msb_encoder(T_msb0)
        T_lsb0, grid_shape_lsb = model.lsb_tokenizer(lsb)
        T_lsb = model.lsb_encoder(T_lsb0)
        assert grid_shape == grid_shape_lsb, "MSB/LSB grids mismatch"
        mask_out = model.mask_head(T_lsb, grid_shape)
        dec_out = model.decoder(x, T_msb, T_lsb, mask_out["m_hat_tok"], grid_shape)

    print("\n=== Tokenizer / Encoder ===")
    print(f"grid_shape: {grid_shape} (N={grid_shape[0]*grid_shape[1]})")
    _describe("T_msb0 (tokenizer)", T_msb0)
    _describe("T_msb (encoder)", T_msb)
    _describe("T_lsb0 (tokenizer)", T_lsb0)
    _describe("T_lsb (encoder)", T_lsb)

    print("\n=== Mask Head ===")
    _describe("m_logits", mask_out["m_logits"])
    _describe("m_hat", mask_out["m_hat"])
    _describe("m_hat_tok", mask_out["m_hat_tok"])

    print("\n=== Decoder Outputs ===")
    _describe("residual_gated", dec_out["residual_gated"])
    _describe("y_hat", dec_out["y_hat"])


if __name__ == "__main__":
    main()
