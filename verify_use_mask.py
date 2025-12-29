import torch
from models.bitplane_former_v1 import BitPlaneFormerV1

def test_use_mask():
    # Test with use_mask=False
    model = BitPlaneFormerV1(
        patch_size=8,
        embed_dim=64,
        num_heads=4,
        msb_depth=2,
        dec_depth=2,
        use_mask=False
    )
    
    # Create dummy inputs
    B, C, H, W = 1, 3, 64, 64
    x = torch.randn(B, C, H, W)
    lsb = torch.randn(B, 18, H, W) # 6 bits * 3 channels
    msb = torch.randn(B, 6, H, W)  # 2 bits * 3 channels
    
    # Forward pass
    out = model(x, lsb=lsb, msb=msb)
    print("Forward pass with use_mask=False successful")
    print("Output keys:", out.keys())

    # Test with use_mask=True (default)
    model_default = BitPlaneFormerV1(
        patch_size=8,
        embed_dim=64,
        num_heads=4,
        msb_depth=2,
        dec_depth=2,
        use_mask=True
    )
    out_default = model_default(x, lsb=lsb, msb=msb)
    print("Forward pass with use_mask=True successful")

if __name__ == "__main__":
    test_use_mask()
