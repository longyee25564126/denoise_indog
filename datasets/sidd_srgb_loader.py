import os
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    """
    Paired noisy/clean loader for SIDD Small sRGB.

    Expects directory structure:
        root_dir/
            Data/
                <scene_dir>/
                    NOISY_SRGB_*.PNG
                    GT_SRGB_*.PNG
    Each noisy file is paired with its corresponding GT file by replacing the prefix "NOISY" with "GT".
    """

    def __init__(self, root_dir: str, split: str = "train", transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split  # kept for interface compatibility; SIDD Small has no explicit splits
        self.transform = transform or transforms.ToTensor()

        data_dir = os.path.join(root_dir, "Data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.samples: list[Tuple[str, str, int]] = []
        for scene_dir in sorted(os.listdir(data_dir)):
            full_dir = os.path.join(data_dir, scene_dir)
            if not os.path.isdir(full_dir):
                continue
            filenames = os.listdir(full_dir)
            noisy_files = sorted(
                [f for f in filenames if f.upper().startswith("NOISY_SRGB_") and f.lower().endswith(".png")]
            )
            if not noisy_files:
                continue
            for noisy_name in noisy_files:
                gt_name = noisy_name.replace("NOISY", "GT", 1)
                noisy_path = os.path.join(full_dir, noisy_name)
                gt_path = os.path.join(full_dir, gt_name)
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"Missing GT file for {noisy_path}: expected {gt_path}")
                label = len(self.samples)  # dummy label; unique per sample
                self.samples.append((noisy_path, gt_path, label))

        if not self.samples:
            raise ValueError(f"No noisy/GT pairs found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        noisy_path, gt_path, label = self.samples[idx]
        noisy = Image.open(noisy_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        noisy = self.transform(noisy)
        gt = self.transform(gt)

        return noisy, gt, label
