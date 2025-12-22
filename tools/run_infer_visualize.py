import argparse
import os
import sys
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def _tokens_to_heatmap(
    tokens: torch.Tensor,
    grid_shape: tuple[int, int],
    out_size: tuple[int, int],
    pad_tokens: int = 0,
) -> torch.Tensor:
    """
    Convert tokens (B, N, D) to a normalized heatmap (B, 1, H, W).
    """
    B, N, _ = tokens.shape
    h, w = grid_shape
    assert N == h * w, f"Token count {N} != grid {h}x{w}"
    mag = tokens.norm(dim=-1).view(B, 1, h, w)
    if pad_tokens > 0:
        mag = mag[:, :, pad_tokens:-pad_tokens, pad_tokens:-pad_tokens]
    mag_min = mag.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag_max = mag.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    mag = F.interpolate(mag, size=out_size, mode="nearest")
    return mag


def compute_mask_gt(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_size: int,
    patch_stride: int,
    temperature: float,
) -> torch.Tensor:
    x_u8 = to_uint8(x)
    y_u8 = to_uint8(y)
    err = (x_u8.to(torch.float32) - y_u8.to(torch.float32)).abs().mean(dim=1, keepdim=True)
    mask_pix = torch.clamp(err / temperature, 0.0, 1.0)
    mask_gt = F.avg_pool2d(mask_pix, kernel_size=patch_size, stride=patch_stride)
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
        patch_stride=args.patch_stride,
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
    parser.add_argument("--patch-stride", type=int, default=None, help="Patch stride for overlapping patches (default: patch_size).")
    parser.add_argument("--pad-size", type=int, default=0, help="Zero padding size applied in the model (pixels).")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--augment", action="store_true", help="Enable augment for loading (usually off for eval).")
    parser.add_argument("--fit-to-patch", action="store_true", help="Center-crop to nearest patch-aligned size.")
    parser.add_argument("--lsb-bits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--msb-bits", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--mask-T", type=float, default=48.0, help="Temperature for residual mask.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help='Device (default: "cuda" if available else "cpu").')
    parser.add_argument("--save-dir", type=str, default="outputs/infer_vis", help="Directory to save visualizations.")
    parser.add_argument("--dec-type", type=str, default=None, choices=["fuse_encoder", "decoder_q_msb", "std_encdec_msb"], help="Decoder variant; if None, try to read from checkpoint.")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt = None
    ckpt_args = {}
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            ckpt_args = ckpt.get("args", {}) or {}
            if args.patch_stride is None and "patch_stride" in ckpt_args:
                args.patch_stride = ckpt_args["patch_stride"]
            if "pad_size" in ckpt_args:
                args.pad_size = ckpt_args["pad_size"]
            if args.dec_type is None and "dec_type" in ckpt_args:
                args.dec_type = ckpt_args["dec_type"]
            if "patch_size" in ckpt_args and ckpt_args["patch_size"] != args.patch_size:
                print(f"Warning: checkpoint patch_size={ckpt_args['patch_size']} != args.patch_size={args.patch_size}")

    if args.patch_stride is None:
        args.patch_stride = args.patch_size

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
        patch_stride=args.patch_stride,
        pad_size=args.pad_size,
        lsb_bits=args.lsb_bits,
        msb_bits=args.msb_bits,
        dec_type=args.dec_type or "fuse_encoder",
    ).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.eval()

    with torch.no_grad():
        out = model({"x": x, "lsb": lsb, "msb": msb})
        mask_gt = compute_mask_gt(x, y, args.patch_size, args.patch_stride, args.mask_T)
        msb_for_vis = msb
        if model.pad_size > 0:
            pad = model.pad_size
            msb_for_vis = F.pad(msb, (pad, pad, pad, pad), mode="constant", value=0.0)
        T_msb0, grid_shape = model.msb_tokenizer(msb_for_vis)
        T_msb = model.msb_encoder(T_msb0)
        pad_tokens = model.pad_size // model.patch_stride if model.pad_size > 0 else 0
        msb_vis = _tokens_to_heatmap(T_msb, grid_shape, x.shape[-2:], pad_tokens=pad_tokens)

    y_hat = out["y_hat"].clamp(0.0, 1.0).squeeze(0)
    x_vis = x.squeeze(0)
    y_vis = y.squeeze(0)
    m_pred = out.get("m_hat", None)
    m_gt = mask_gt.cpu().squeeze(0)
    msb_map = msb_vis.cpu().squeeze(0)

    # Build a labeled grid with equal-sized cells.
    imgs = [
        ("noisy", tensor_to_pil(x_vis)),
        ("clean", tensor_to_pil(y_vis)),
        ("denoised", tensor_to_pil(y_hat)),
        ("msb_enc", tensor_to_pil(msb_map)),
    ]
    if m_pred is not None:
        imgs.append(("mask_pred", tensor_to_pil(m_pred.detach().cpu().squeeze(0))))
    imgs.append(("mask_gt", tensor_to_pil(m_gt)))

    target_w, target_h = imgs[0][1].size
    resized = []
    for name, img in imgs:
        is_mask = name.startswith("mask")
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (target_w, target_h):
            resample = Image.NEAREST if is_mask else Image.BILINEAR
            img = img.resize((target_w, target_h), resample=resample)
        resized.append((name, img))

    font = ImageFont.load_default()
    tmp = Image.new("RGB", (10, 10), color=(255, 255, 255))
    draw_tmp = ImageDraw.Draw(tmp)
    header_h = max(_text_size(draw_tmp, name, font)[1] for name, _ in resized) + 8

    cols = len(resized)
    grid_w = cols * target_w
    grid_h = header_h + target_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, (name, img) in enumerate(resized):
        x0 = i * target_w
        tw, th = _text_size(draw, name, font)
        tx = x0 + (target_w - tw) // 2
        ty = (header_h - th) // 2
        draw.text((tx, ty), name, fill=(0, 0, 0), font=font)
        grid.paste(img, (x0, header_h))

    os.makedirs(args.save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(sample.get("path_noisy") or f"idx_{idx}"))[0]
    grid_path = os.path.join(args.save_dir, f"{stem}_viz.png")
    grid.save(grid_path)

    print(f"Dataset size {info['size']}, visualized index {idx}")
    print(f"Saved grid: {grid_path}")
    if m_pred is not None:
        mp = m_pred.detach().cpu()
        print(
            f"mask_pred mean/std: {mp.mean().item():.4f}/{mp.std().item():.4f} | "
            f"mask_gt mean/std: {m_gt.mean().item():.4f}/{m_gt.std().item():.4f}"
        )
    else:
        print(f"mask_gt mean/std: {m_gt.mean().item():.4f}/{m_gt.std().item():.4f}")


if __name__ == "__main__":
    main()
