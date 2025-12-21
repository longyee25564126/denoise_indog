import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bitplane_utils import (
    lsb_xor_mask,
    make_lsb_msb,
    pool_to_patch_mask,
    to_uint8,
    residual_soft_mask,
)


def _load_image_as_tensor(path: str) -> torch.Tensor:
    """
    Load an image file and return a float tensor in [0, 1] with shape (3, H, W).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # H x W x 3
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor


def _random_crop_pair(
    img_a: torch.Tensor, img_b: torch.Tensor, crop_size: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if crop_size is None:
        return img_a, img_b
    _, h, w = img_a.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size {crop_size} exceeds image size {(h, w)}")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    slc = (slice(top, top + crop_size), slice(left, left + crop_size))
    return img_a[:, slc[0], slc[1]], img_b[:, slc[0], slc[1]]


def _augment_pair(img_a: torch.Tensor, img_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        img_a = torch.flip(img_a, dims=[2])  # horizontal flip
        img_b = torch.flip(img_b, dims=[2])
    if random.random() < 0.5:
        img_a = torch.flip(img_a, dims=[1])  # vertical flip
        img_b = torch.flip(img_b, dims=[1])
    k = random.randint(0, 3)
    if k:
        img_a = torch.rot90(img_a, k=k, dims=[1, 2])
        img_b = torch.rot90(img_b, k=k, dims=[1, 2])
    return img_a, img_b


class BitPlanePairDataset(Dataset):
    """
    Paired noisy/clean dataset that returns bit-planes and patch-level mask.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: str,
        pairs_file: Optional[str] = None,
        patch_size: int = 8,
        patch_stride: int | None = None,
        crop_size: Optional[int] = None,
        augment: bool = False,
        strict_pairing: bool = True,
        return_mask_flat: bool = False,
        use_residual_mask: bool = True,
        mask_temperature: float = 48.0,
        mask_use_quantile: bool = False,
        mask_quantile: float = 0.9,
        lsb_bits: tuple[int, int] | list[int] = (0, 5),
        msb_bits: tuple[int, int] | list[int] = (6, 7),
    ):
        super().__init__()
        self.root = root
        self.pairs_file = pairs_file
        self.patch_size = patch_size
        if patch_stride is None:
            patch_stride = patch_size
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if patch_stride > patch_size:
            raise ValueError("patch_stride should not exceed patch_size (would create gaps)")
        self.patch_stride = patch_stride
        self.crop_size = crop_size
        self.augment = augment
        self.strict_pairing = strict_pairing
        self.return_mask_flat = return_mask_flat
        self.use_residual_mask = use_residual_mask
        self.mask_temperature = mask_temperature
        self.mask_use_quantile = mask_use_quantile
        self.mask_quantile = mask_quantile
        self.lsb_bits = lsb_bits
        self.msb_bits = msb_bits

        if self.crop_size is not None:
            if self.crop_size < self.patch_size:
                raise ValueError("crop_size must be >= patch_size")
            if (self.crop_size - self.patch_size) % self.patch_stride != 0:
                raise ValueError("crop_size must align with patch_size/patch_stride")

        if pairs_file:
            self.samples = self._load_from_pairs_file(pairs_file)
        else:
            self.samples = self._scan_root_for_pairs(root)

        if not self.samples:
            raise ValueError("No paired samples found.")

    def _load_from_pairs_file(self, pairs_file: str) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        with open(pairs_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    if self.strict_pairing:
                        raise ValueError(f"Invalid line in pairs file: {line}")
                    continue
                noisy_path, clean_path = parts
                if not os.path.isabs(noisy_path):
                    noisy_path = os.path.join(self.root, noisy_path)
                if not os.path.isabs(clean_path):
                    clean_path = os.path.join(self.root, clean_path)
                if os.path.exists(noisy_path) and os.path.exists(clean_path):
                    samples.append((noisy_path, clean_path))
                elif self.strict_pairing:
                    raise FileNotFoundError(f"Missing file in pairs: {noisy_path}, {clean_path}")
        return samples

    def _scan_root_for_pairs(self, root: str) -> List[Tuple[str, str]]:
        noisy_root = os.path.join(root, "noisy")
        clean_root = os.path.join(root, "clean")
        samples: List[Tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(noisy_root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in self.IMG_EXTS:
                    continue
                noisy_path = os.path.join(dirpath, name)
                rel = os.path.relpath(noisy_path, noisy_root)
                clean_path = os.path.join(clean_root, rel)
                if os.path.exists(clean_path):
                    samples.append((noisy_path, clean_path))
                elif self.strict_pairing:
                    raise FileNotFoundError(f"Missing clean file for {rel}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        noisy_path, clean_path = self.samples[idx]
        x = _load_image_as_tensor(noisy_path)  # float [0,1]
        y = _load_image_as_tensor(clean_path)

        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch for pair: {noisy_path}, {clean_path}")

        x, y = _random_crop_pair(x, y, self.crop_size)

        if self.augment:
            x, y = _augment_pair(x, y)

        _, H, W = x.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image size {(H, W)} smaller than patch_size {self.patch_size}")
        if (H - self.patch_size) % self.patch_stride != 0 or (W - self.patch_size) % self.patch_stride != 0:
            raise ValueError(
                f"Image size {(H, W)} not aligned to patch_size/patch_stride "
                f"(P={self.patch_size}, S={self.patch_stride})"
            )

        x_u8 = to_uint8(x)
        y_u8 = to_uint8(y)

        lsb, msb = make_lsb_msb(x_u8, lsb_bits=self.lsb_bits, msb_bits=self.msb_bits)
        if self.use_residual_mask:
            mask_gt = residual_soft_mask(
                x,
                y,
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                temperature=self.mask_temperature,
                use_quantile=self.mask_use_quantile,
                quantile=self.mask_quantile,
            )
        else:
            mask_pix = lsb_xor_mask(x_u8, y_u8)
            mask_gt = pool_to_patch_mask(mask_pix, self.patch_size, self.patch_stride)
        if self.return_mask_flat:
            mask_gt = mask_gt.flatten().unsqueeze(1)  # (h*w, 1); grid shape is (1, h, w) by default

        sample = {
            "x": x.to(torch.float32),
            "y": y.to(torch.float32),
            "lsb": lsb,
            "msb": msb,
            "mask_gt": mask_gt,
            "path_noisy": noisy_path,
            "path_clean": clean_path,
        }
        return sample


# DataLoader usage example:
# ds = BitPlanePairDataset(root="/path/to/data", patch_size=8, crop_size=256, augment=True)
# dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
# batch = next(iter(dl))
