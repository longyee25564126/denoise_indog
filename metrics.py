import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2)**2 / float(2 * sigma**2)))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the SSIM between two images.
    img1, img2: (B, C, H, W) tensors in range [0, 1]
    """
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        # Load VGG16 pretrained on ImageNet
        vgg = models.vgg16(pretrained=True)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        # Split VGG into slices
        for x in range(4):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg.features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg.features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg.features[x])
            
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Input and target should be in [0, 1]
        # Normalize to [-1, 1]? No, VGG expects normalized by mean/std
        
        # Normalize inputs
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        h_input = input
        h_target = target
        
        outs_input = []
        outs_target = []
        
        h_input = self.slice1(h_input)
        h_target = self.slice1(h_target)
        outs_input.append(h_input)
        outs_target.append(h_target)
        
        h_input = self.slice2(h_input)
        h_target = self.slice2(h_target)
        outs_input.append(h_input)
        outs_target.append(h_target)
        
        h_input = self.slice3(h_input)
        h_target = self.slice3(h_target)
        outs_input.append(h_input)
        outs_target.append(h_target)
        
        h_input = self.slice4(h_input)
        h_target = self.slice4(h_target)
        outs_input.append(h_input)
        outs_target.append(h_target)
        
        h_input = self.slice5(h_input)
        h_target = self.slice5(h_target)
        outs_input.append(h_input)
        outs_target.append(h_target)
        
        diff = 0
        for k in range(len(outs_input)):
            diff += torch.mean((outs_input[k] - outs_target[k]) ** 2)
            
        return diff
