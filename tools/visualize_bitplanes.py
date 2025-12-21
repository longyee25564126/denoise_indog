import argparse
import math
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from datasets.bitplane_dataset import BitPlanePairDataset
from datasets.bitplane_utils import to_uint8
from datasets.external_adapter import ExternalPairedBitPlaneDataset


def _compute_bit_planes(x_rgb: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Returns:
    - 8 RGB bit-plane maps (each is 3 x H x W, uint8 0/255, showing the bit per channel)
    - 24 single-channel bit-planes (R/G/B each 8 bits, uint8 0/255) for detailed view
    - the uint8 RGB image
    """
    u8 = to_uint8(x_rgb)  # (3, H, W), uint8

    planes_rgb_bits: list[torch.Tensor] = []
    for bit in range(8):
        plane = ((u8 >> bit) & 1) * 255  # 3 x H x W, uint8
        planes_rgb_bits.append(plane.to(torch.uint8))

    planes_per_channel: list[torch.Tensor] = []
    for c in range(3):  # R,G,B
        for bit in range(8):
            plane = ((u8[c] >> bit) & 1) * 255  # H x W, uint8 0 or 255
            planes_per_channel.append(plane.to(torch.uint8))

    return planes_rgb_bits, planes_per_channel, u8


def _make_grid(
    bit_planes: list[torch.Tensor],
    labels: list[str],
    scale: int = 1,
    cols: int = 4,
    header_colors: list | None = None,
    text_colors: list | None = None,
) -> Image.Image:
    """
    Compose bit-plane images (either HxW or 3xHxW) into a grid.

    Each cell is rendered like a table cell: a header row with the label, followed by the bit-plane image.
    """
    first = bit_planes[0]
    if first.ndim == 2:
        h, w = first.shape
        mode = "L"
    elif first.ndim == 3:
        _, h, w = first.shape
        mode = "RGB"
    else:
        raise ValueError("Unsupported bit-plane tensor shape")

    cell_w, cell_h = w * scale, h * scale
    header_h = max(14, int(0.08 * cell_h))  # header height in pixels
    gap = max(4, int(0.03 * min(cell_w, cell_h)))  # spacing between cells
    cols = max(1, cols)
    rows = math.ceil(len(bit_planes) / cols)
    canvas_w = cols * cell_w + (cols - 1) * gap
    canvas_h = rows * (cell_h + header_h) + (rows - 1) * gap
    # Use RGB canvas if we need colored headers/text
    canvas_mode = mode
    if header_colors is not None or text_colors is not None:
        canvas_mode = "RGB"
    bg_canvas = 255 if canvas_mode == "L" else (255, 255, 255)
    canvas = Image.new(canvas_mode, (canvas_w, canvas_h), color=bg_canvas)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def _norm_color(color, target_mode):
        if target_mode == "RGB":
            if isinstance(color, tuple):
                return color
            return (color, color, color)
        else:  # "L"
            if isinstance(color, tuple):
                return color[0]
            return color

    for idx, plane in enumerate(bit_planes):
        if plane.ndim == 3:
            arr = plane.permute(1, 2, 0).cpu().numpy()
        else:
            arr = plane.cpu().numpy()
        img = Image.fromarray(arr, mode=mode)
        if canvas_mode == "RGB" and img.mode == "L":
            img = img.convert("RGB")
        if scale != 1:
            img = img.resize((cell_w, cell_h), Image.NEAREST)
        r, c = divmod(idx, cols)
        x0 = c * (cell_w + gap)
        y0 = r * (cell_h + header_h + gap)
        # Header background
        text = labels[idx]
        default_text = 0 if canvas_mode == "L" else (0, 0, 0)
        default_bg = 255 if canvas_mode == "L" else (255, 255, 255)
        text_color = _norm_color(
            text_colors[idx] if text_colors and idx < len(text_colors) else default_text,
            canvas_mode,
        )
        bg_color = _norm_color(
            header_colors[idx] if header_colors and idx < len(header_colors) else default_bg,
            canvas_mode,
        )
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=bg_color)
        # textbbox is available in modern Pillow; fallback to approximate if missing
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)  # type: ignore[attr-defined]
        tx = x0 + (cell_w - tw) // 2
        ty = y0 + (header_h - th) // 2
        draw.text((tx, ty), text, fill=text_color, font=font)
        # Paste image below header
        canvas.paste(img, (x0, y0 + header_h))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 8 bit-planes for a random dataset image.")
    parser.add_argument("--root", type=str, required=True, help="Root containing noisy/clean data.")
    parser.add_argument("--pairs-file", type=str, default=None, help="Optional tab-separated pairs file (internal loader).")
    parser.add_argument("--use-external", action="store_true", help="Use ExternalPairedBitPlaneDataset.")
    parser.add_argument("--external-module", type=str, default="/home/longyee/datasets/dataset_and_data_loader/data_loader.py", help="Path to external data_loader.py defining PairedImageDataset.")
    parser.add_argument("--split", type=str, default="train", help="Split name for external loader.")
    parser.add_argument("--patch-size", type=int, default=8, help="Patch size used by dataset checks.")
    parser.add_argument("--patch-stride", type=int, default=None, help="Patch stride for overlapping patches (default: patch_size).")
    parser.add_argument("--crop-size", type=int, default=None, help="Optional crop size (must divide patch size).")
    parser.add_argument("--index", type=int, default=None, help="Optional fixed sample index; defaults to random.")
    parser.add_argument("--scale", type=int, default=1, help="Integer up/down-scale when composing the grid.")
    parser.add_argument("--per-channel", action="store_true", help="Also save per-channel (R/G/B) bit-planes.")
    args = parser.parse_args()
    if args.patch_stride is None:
        args.patch_stride = args.patch_size

    if args.use_external:
        ds = ExternalPairedBitPlaneDataset(
            module_path=args.external_module,
            root_dir=args.root,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            crop_size=args.crop_size,
            augment=False,
            return_mask_flat=False,
            split=args.split,
        )
    else:
        ds = BitPlanePairDataset(
            root=args.root,
            pairs_file=args.pairs_file,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            crop_size=args.crop_size,
            augment=False,
            strict_pairing=True,
            return_mask_flat=False,
        )

    idx = args.index if args.index is not None else random.randrange(len(ds))
    sample = ds[idx]
    planes_rgb_bits, planes_per_channel, u8_rgb = _compute_bit_planes(sample["x"])

    labels = [f"bit {b}" for b in range(8)]
    grid = _make_grid(planes_rgb_bits, labels, scale=max(1, args.scale), cols=4)

    rgb_img = Image.fromarray(u8_rgb.permute(1, 2, 0).cpu().numpy(), mode="RGB")

    stem = os.path.splitext(os.path.basename(sample.get("path_noisy") or f"idx_{idx}"))[0]
    out_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, f"{stem}_bitplanes.png")
    rgb_path = os.path.join(out_dir, f"{stem}_rgb.png")

    grid.save(grid_path)
    rgb_img.save(rgb_path)

    print(f"Sample idx: {idx}, noisy path: {sample.get('path_noisy')}")
    print(f"Saved bit-plane grid (RGB per bit) to: {grid_path}")
    print(f"Saved RGB reference to: {rgb_path}")

    if args.per_channel:
        ch_labels = [f"{c} b{b}" for c in ["R", "G", "B"] for b in range(8)]
        # Color-coded headers/text for R/G/B
        color_map = {
            "R": ((255, 220, 220), (200, 0, 0)),
            "G": ((220, 255, 220), (0, 140, 0)),
            "B": ((220, 230, 255), (0, 0, 200)),
        }
        header_colors = [color_map[label[0]][0] for label in ch_labels]
        text_colors = [color_map[label[0]][1] for label in ch_labels]
        grid_ch = _make_grid(
            planes_per_channel,
            ch_labels,
            scale=max(1, args.scale),
            cols=8,
            header_colors=header_colors,
            text_colors=text_colors,
        )
        grid_ch_path = os.path.join(out_dir, f"{stem}_bitplanes_rgb.png")
        grid_ch.save(grid_ch_path)
        print(f"Saved per-channel bit-planes to: {grid_ch_path}")


if __name__ == "__main__":
    main()
