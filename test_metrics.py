import torch
from metrics import LPIPS, ssim

def test_metrics():
    print("Testing metrics implementation...")
    
    # Create dummy images (B, C, H, W)
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    
    # Test SSIM
    try:
        ssim_val = ssim(img1, img2)
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
