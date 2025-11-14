import torch
import torch.nn as nn
from conv_block import ConvBlock
from upconv_block import UpConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_filters=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters*2)
        self.enc3 = ConvBlock(base_filters*2, base_filters*4)
        self.enc4 = ConvBlock(base_filters*4, base_filters*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_filters*8, base_filters*16)

        self.up4 = UpConvBlock(base_filters*16, base_filters*8)
        self.up3 = UpConvBlock(base_filters*8, base_filters*4)
        self.up2 = UpConvBlock(base_filters*4, base_filters*2)
        self.up1 = UpConvBlock(base_filters*2, base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        out = self.final(d1)
        return out
