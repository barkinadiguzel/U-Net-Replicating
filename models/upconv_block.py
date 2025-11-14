import torch
import torch.nn as nn
from conv_block import ConvBlock

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
