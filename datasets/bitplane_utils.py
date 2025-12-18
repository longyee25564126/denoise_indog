import torch


def to_uint8(img_float01: torch.Tensor) -> torch.Tensor:
    """
    Convert a float image in [0, 1] with shape (3, H, W) to uint8.
    """
    return torch.clamp(torch.round(img_float01 * 255.0), 0, 255).to(torch.uint8)


def extract_bit_planes(u8_img: torch.Tensor, bits: list[int]) -> torch.Tensor:
    """
    Extract bit-planes from a uint8 image.

    Args:
        u8_img: Tensor uint8 with shape (3, H, W).
        bits: List of bit indices to extract.

    Returns:
        Float tensor with shape (3 * len(bits), H, W) containing 0/1 values.
    """
    planes = [((u8_img >> k) & 1) for k in bits]
    stacked = torch.stack(planes, dim=0)  # nbits x 3 x H x W
    nbits, c, h, w = stacked.shape
    return stacked.view(nbits * c, h, w).to(torch.float32)


def make_lsb_msb(u8_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build LSB and MSB stacks from a uint8 image.

    Returns:
        lsb: bits [0,1,2,3], shape (12, H, W)
        msb: bits [4,5,6,7], shape (12, H, W)
    """
    lsb = extract_bit_planes(u8_img, [0, 1, 2, 3])
    msb = extract_bit_planes(u8_img, [4, 5, 6, 7])
    return lsb, msb


def lsb_xor_mask(u8_noisy: torch.Tensor, u8_clean: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel LSB XOR mask averaged over RGB and bits.

    Args:
        u8_noisy: uint8 (3, H, W)
        u8_clean: uint8 (3, H, W)

    Returns:
        Float32 mask with shape (1, H, W) in [0, 1].
    """
    acc = torch.zeros_like(u8_noisy[0], dtype=torch.float32)
    for k in range(4):
        xor_k = ((u8_noisy >> k) & 1) ^ ((u8_clean >> k) & 1)  # 3 x H x W, values 0/1
        acc = acc + xor_k.float().mean(dim=0)  # H x W
    m_gt_pix = (acc / 4.0).unsqueeze(0)  # 1 x H x W
    return m_gt_pix


def pool_to_patch_mask(m_gt_pix: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Average-pool pixel-level mask to patch grid.

    Args:
        m_gt_pix: float tensor (1, H, W)
        patch_size: pooling kernel/stride

    Returns:
        mask_gt_grid: float tensor (1, h, w)
    """
    import torch.nn.functional as F

    pooled = F.avg_pool2d(m_gt_pix.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
    return pooled.squeeze(0)
