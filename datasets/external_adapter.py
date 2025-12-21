"""
Adapter to wrap an external PairedImageDataset (noisy, clean, label) and
produce bit-plane tensors + patch-level mask compatible with BitPlaneFormer.
"""
import importlib.util
import os
import random
from types import ModuleType
from typing import Optional

import torch
from torch.utils.data import Dataset

from .bitplane_utils import (
    lsb_xor_mask,
    make_lsb_msb,
    pool_to_patch_mask,
    to_uint8,
    residual_soft_mask,
)


def _load_module(module_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("external_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_crop_pair(
    img_a: torch.Tensor, img_b: torch.Tensor, crop_size: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size is None:
        return img_a, img_b
    _, h, w = img_a.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size {crop_size} exceeds image size {(h, w)}")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    slc = (slice(top, top + crop_size), slice(left, left + crop_size))
    return img_a[:, slc[0], slc[1]], img_b[:, slc[0], slc[1]]


def _augment_pair(img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class ExternalPairedBitPlaneDataset(Dataset):
    """
    Wrap an external PairedImageDataset (noisy, clean, label) and add bit-plane/mask outputs.

    Expected external dataset signature: dataset[idx] -> (noisy, clean, label)
    where noisy/clean are float tensors in [0,1], shape (3,H,W).
    """

    def __init__(
        self,
        module_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        base_dataset: Optional[Dataset] = None,
        patch_size: int = 8,
        patch_stride: int | None = None,
        crop_size: Optional[int] = None,
        augment: bool = False,
        return_mask_flat: bool = False,
        split: str = "train",
        fit_to_patch: bool = False,
        use_residual_mask: bool = True,
        mask_temperature: float = 48.0,
        mask_use_quantile: bool = False,
        mask_quantile: float = 0.9,
        lsb_bits: tuple[int, int] | list[int] = (0, 5),
        msb_bits: tuple[int, int] | list[int] = (6, 7),
    ):
        super().__init__()
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
        self.return_mask_flat = return_mask_flat
        self.split = split
        self.fit_to_patch = fit_to_patch
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

        if base_dataset is not None:
            self.base_dataset = base_dataset
        else:
            if module_path is None or root_dir is None:
                raise ValueError("Either base_dataset must be provided or module_path/root_dir must be specified")
            module = _load_module(module_path)
            if not hasattr(module, "PairedImageDataset"):
                raise AttributeError(f"Module {module_path} has no PairedImageDataset")
            PairedImageDataset = getattr(module, "PairedImageDataset")
            try:
                self.base_dataset = PairedImageDataset(root_dir=root_dir, split=split, transform=None)
            except TypeError:
                # Fallback if external class does not support split
                self.base_dataset = PairedImageDataset(root_dir=root_dir, transform=None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        noisy, clean, label = self.base_dataset[idx]
        noisy = noisy.to(torch.float32)
        clean = clean.to(torch.float32)

        if noisy.shape != clean.shape:
            raise ValueError(f"Shape mismatch for pair idx {idx}: {noisy.shape} vs {clean.shape}")

        noisy, clean = _random_crop_pair(noisy, clean, self.crop_size)
        if self.augment:
            noisy, clean = _augment_pair(noisy, clean)

        if self.fit_to_patch:
            noisy = self._fit_to_patch(noisy)
            clean = self._fit_to_patch(clean)

        _, H, W = noisy.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image size {(H, W)} smaller than patch_size {self.patch_size}")
        if (H - self.patch_size) % self.patch_stride != 0 or (W - self.patch_size) % self.patch_stride != 0:
            raise ValueError(
                f"Image size {(H, W)} not aligned to patch_size/patch_stride "
                f"(P={self.patch_size}, S={self.patch_stride})"
            )

        noisy_u8 = to_uint8(noisy)
        clean_u8 = to_uint8(clean)

        lsb, msb = make_lsb_msb(noisy_u8, lsb_bits=self.lsb_bits, msb_bits=self.msb_bits)
        if self.use_residual_mask:
            mask_gt = residual_soft_mask(
                noisy,
                clean,
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                temperature=self.mask_temperature,
                use_quantile=self.mask_use_quantile,
                quantile=self.mask_quantile,
            )
        else:
            mask_pix = lsb_xor_mask(noisy_u8, clean_u8)
            mask_gt = pool_to_patch_mask(mask_pix, self.patch_size, self.patch_stride)
        if self.return_mask_flat:
            mask_gt = mask_gt.flatten().unsqueeze(1)  # (h*w, 1)

        # Try to include paths if the external dataset stores them.
        path_noisy, path_clean = None, None
        samples = getattr(self.base_dataset, "samples", None)
        if samples is not None and len(samples) > idx:
            path_noisy = samples[idx][0]
            path_clean = samples[idx][1]

        return {
            "x": noisy,
            "y": clean,
            "lsb": lsb,
            "msb": msb,
            "mask_gt": mask_gt,
            "label": label,
            "path_noisy": path_noisy,
            "path_clean": path_clean,
        }

    def _fit_to_patch(self, img: torch.Tensor) -> torch.Tensor:
        """
        Center-crop to the largest region aligned to patch_size/patch_stride.
        """
        _, H, W = img.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image too small to fit patch_size {self.patch_size}: {(H, W)}")
        new_h = self.patch_size + ((H - self.patch_size) // self.patch_stride) * self.patch_stride
        new_w = self.patch_size + ((W - self.patch_size) // self.patch_stride) * self.patch_stride
        if new_h == 0 or new_w == 0:
            raise ValueError(f"Image too small to fit patch_size {self.patch_size}: {(H, W)}")
        if new_h == H and new_w == W:
            return img
        top = (H - new_h) // 2
        left = (W - new_w) // 2
        return img[:, top : top + new_h, left : left + new_w]
