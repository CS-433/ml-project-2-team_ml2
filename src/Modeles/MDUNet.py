""" Full assembly of the parts to form the complete network """
import torch
from .UNet_parts import *


class Dilated(nn.Module):
    """Dilating block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dil1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=8),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=16)
        )
        self.dil2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=8)
        )
        self.dil3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=4)
        )
        self.dil4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=2)
        )
        self.dil5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False, dilation=1)
        )

        self.conv = nn.Conv2d(5*out_channels, out_channels, kernel_size=1, padding="same", bias=False)

    def forward(self, x):
        x1 = self.dil1(x)
        x2 = self.dil2(x)
        x3 = self.dil3(x)
        x4 = self.dil4(x)
        x5 = self.dil5(x)
        res = torch.cat([x1, x2, x3, x4, x5], dim=1)
        res = self.conv(res)
        return res


class MDUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MDUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.DIL = Dilated(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.DIL(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        logits = self.sigmoid(logits)
        return logits