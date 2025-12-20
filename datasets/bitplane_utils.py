from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def to_uint8(img_float01: torch.Tensor) -> torch.Tensor:
    """
    Convert a float image in [0, 1] with shape (3, H, W) to uint8.
    """
    return torch.clamp(torch.round(img_float01 * 255.0), 0, 255).to(torch.uint8)


def expand_bits(bits: Sequence[int] | tuple[int, int]) -> list[int]:
    """
    Normalize bit specification into a sorted unique list.
    Accepts:
        - iterable of ints
        - tuple (start, end) inclusive
    """
    if isinstance(bits, tuple) and len(bits) == 2:
        start, end = bits
        bit_list = list(range(start, end + 1))
    else:
        bit_list = list(bits)
    bit_list = sorted(set(int(b) for b in bit_list))
    return bit_list


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


def make_lsb_msb(
    u8_img: torch.Tensor,
    lsb_bits: Sequence[int] | tuple[int, int] = (0, 5),
    msb_bits: Sequence[int] | tuple[int, int] = (6, 7),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build LSB and MSB stacks from a uint8 image, with configurable bit ranges.

    Args:
        u8_img: uint8 tensor (3,H,W)
        lsb_bits: iterable of bit indices or (start,end) inclusive (default 0-5)
        msb_bits: iterable of bit indices or (start,end) inclusive (default 6-7)

    Returns:
        lsb: shape (3 * n_lsb_bits, H, W)
        msb: shape (3 * n_msb_bits, H, W)
    """
    lsb_list = expand_bits(lsb_bits)
    msb_list = expand_bits(msb_bits)
    lsb = extract_bit_planes(u8_img, lsb_list)
    msb = extract_bit_planes(u8_img, msb_list)
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


def residual_soft_mask(
    x_float01: torch.Tensor,
    y_float01: torch.Tensor,
    patch_size: int,
    temperature: float = 32.0,
    use_quantile: bool = False,
    quantile: float = 0.9,
) -> torch.Tensor:
    """
    Build a soft mask from absolute RGB residuals (uint8) pooled to patch grid.

    Args:
        x_float01: noisy image float [0,1], shape (3,H,W) or (B,3,H,W)
        y_float01: clean image float [0,1], shape (3,H,W) or (B,3,H,W)
        patch_size: pooling kernel/stride
        temperature: fixed scale for normalization when use_quantile=False
        use_quantile: if True, use per-image quantile of residual as scale
        quantile: quantile value in (0,1) when use_quantile=True

    Returns:
        mask_gt: float tensor shape (1,h,w) or (B,1,h,w) depending on input dims
    """
    added_batch = False
    if x_float01.ndim == 3:
        x_float01 = x_float01.unsqueeze(0)
        y_float01 = y_float01.unsqueeze(0)
        added_batch = True

    # Use float precision directly, avoid to_uint8 quantization
    # x_float01 and y_float01 are already in [0, 1]
    err = (x_float01 - y_float01).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
    
    # Scale error to be roughly compatible with previous 0-255 scale logic if needed,
    # or just adjust temperature.
    # Previous logic: err_u8 = abs(x*255 - y*255) = abs(x-y)*255
    # So err_u8 = err_float * 255.
    # To keep temperature meaning similar, we can scale err by 255 here.
    err = err * 255.0

    if use_quantile:
        # per-image quantile as scale, avoid zeros
        q = torch.quantile(err.view(err.shape[0], -1), quantile, dim=1, keepdim=True)
        scale = torch.clamp(q, min=1.0).view(-1, 1, 1, 1)
    else:
        scale = torch.tensor(temperature, device=err.device, dtype=err.dtype)

    mask_pix = torch.clamp(err / scale, 0.0, 1.0)
    mask_gt = F.avg_pool2d(mask_pix, kernel_size=patch_size, stride=patch_size)

    if added_batch:
        mask_gt = mask_gt.squeeze(0)
    return mask_gt
