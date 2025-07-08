import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import Normalize

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + residual
        return out

class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_residual_blocks: int = 16):
        super(Generator, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(n_residual_blocks)]
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.residual_blocks(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out = self.upsample_blocks(out)
        out = self.conv_block3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(Discriminator, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.dense(out)
        return out

class VGGLoss(nn.Module):
    def __init__(self, device: torch.device):
        super(VGGLoss, self).__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[:35].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_vgg = self.vgg(self.norm(x))
        y_vgg = self.vgg(self.norm(y))
        loss = self.mse(x_vgg, y_vgg)
        return loss