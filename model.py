import torch.nn as nn
import torchvision.models
from torchvision.transforms import Normalize

class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + residual

        return out

class SRResNet(nn.Module):

    def __init__(self):
        super(SRResNet, self).__init__()
        self.conv_block1 = nn.Sequential(

            nn.Conv2d(kernel_size=9, in_channels=3, out_channels=64, stride=1, padding=4),
            nn.PReLU()

        )
        self.residual_blocks = nn.Sequential(

            *[ResidualBlock() for _ in range(16)]

        )
        self.conv_block2 = nn.Sequential(

            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
            nn.BatchNorm2d(64)

        )
        self.upsample_blocks = nn.Sequential(

            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=256, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=256, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(kernel_size=9, in_channels=64, out_channels=3, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.conv_block1(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv_block2(out)
        out = out + residual
        out = self.upsample_blocks(out)
        out = self.conv_block3(out)

        return out


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.generator = SRResNet()

    def forward(self, x):

        output = self.generator(x)

        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=64, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(

            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.dense = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):

        output = self.conv_block1(x)
        output = self.conv_block2(output)
        output = self.adaptive_pool(output)
        output = self.dense(output.view(x.size(0), -1))

        return output

class VGG54(nn.Module):

    def __init__(self):
    
        super(VGG54, self).__init__()
        
        self.truncated_vgg54 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[:35].eval()
    
        for _, param in self.truncated_vgg54.named_parameters():
            param.requires_grad = False

    def forward(self, x):

        x = self.truncated_vgg54(x)

        return x

class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()

        self.vgg = VGG54()
        self.mse = nn.MSELoss()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):

        x = self.vgg(self.norm(x))
        y = self.vgg(self.norm(y))

        loss = self.mse(x, y)

        return loss