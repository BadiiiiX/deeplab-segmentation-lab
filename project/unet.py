import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    A lightweight UNet-like architecture to upsample an 8x8 radio map to a 64x64 segmentation mask.
    Includes skip connections and uses transposed convolutions for better detail reconstruction.
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # skip from enc2 (32)
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # skip from enc1 (16)
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.final = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # -> (B, 16, 8, 8)
        x2 = self.enc2(x1)  # -> (B, 32, 8, 8)

        # Bottleneck
        x = self.bottleneck(x2)  # -> (B, 64, 8, 8)

        # Decoder with skips
        x = self.up1(x)  # -> (B, 32, 16, 16)
        x = torch.cat([x, nn.functional.interpolate(x2, size=(16, 16), mode='nearest')], dim=1)
        x = self.dec1(x)

        x = self.up2(x)  # -> (B, 16, 32, 32)
        x = torch.cat([x, nn.functional.interpolate(x1, size=(32, 32), mode='nearest')], dim=1)
        x = self.dec2(x)

        x = self.up3(x)  # -> (B, 8, 64, 64)
        x = self.final(x)  # -> (B, 1, 64, 64)

        return x
