import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """3-level U-Net architecture for face restoration."""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

         
        self.e1 = Block(3, 64)
        self.e2 = Block(64, 128)
        self.e3 = Block(128, 256)

         
        self.bottleneck = Block(256, 512)

         
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = Block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = Block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = Block(128, 64)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))
