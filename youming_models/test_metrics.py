import torch
import sys
import os

# Add root to sys.path to allow imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from youming_models.metrics import calculate_psnr, calculate_ssim, LPIPS

def test_metrics():
    print("Testing youming_models metrics implementation...")
    
    # Create dummy images (B, C, H, W)
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    
    # Test PSNR
    try:
        psnr_val = calculate_psnr(img1, img2)
        print(f"PSNR: {psnr_val:.4f}")
    except Exception as e:
        print(f"PSNR failed: {e}")

    # Test SSIM
    try:
        ssim_val = calculate_ssim(img1, img2)
        print(f"SSIM: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"SSIM failed: {e}")
        
    # Test LPIPS
    try:
        lpips_model = LPIPS()
        lpips_val = lpips_model(img1, img2)
        print(f"LPIPS: {lpips_val.item():.4f}")
    except Exception as e:
        print(f"LPIPS failed: {e}")
        
if __name__ == "__main__":
    test_metrics()
