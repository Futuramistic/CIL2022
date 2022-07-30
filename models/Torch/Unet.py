"""
U-Network, adjusted from https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Performs two consecutive convolutions
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int): If specified, indicates the number of channels between the 2 convolutions,
            else, it is automatically set to the number of output channels
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): if bilinear, use the normal convolutions with upscaling to reduce the number of channels,
            else perform a transposed convolution
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, torch.div(out_channels, 2, rounding_mode='trunc'),
                                   torch.div(in_channels, 2, rounding_mode='trunc'))
        else:
            self.up = nn.ConvTranspose2d(in_channels , torch.div(in_channels, 2, rounding_mode='trunc'),
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Upscale x1 and concatenate it with x2
        Args:
            x1 (tensor): tensor from previous block in the upscaling path
            x2 (tensor): tensor transferred from the encoding path via a skip-connection
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'), torch.div(diffX - diffX, 2, rounding_mode='trunc'),
                        torch.div(diffY, 2, rounding_mode='trunc'), torch.div(diffY - diffY, 2, rounding_mode='trunc')])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Perform the final convolution of the network (A simple 2D 1x1 convolution)
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x


class UNet(nn.Module):
    """
    Original UNet architecture
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        """
        Args:
            n_channels (int): Number of expected input channels (typically 3)
            n_classes (int): Dimensionality of output, corresponds to the number of classes in the dataset
            bilinear (bool): if bilinear, use the normal convolutions with upscaling to reduce the number of channels,
            else perform a transposed convolution in the decoding part of the network (i.e. bilinear=False is more
            parameter-heavy)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)
        self.activation = nn.Softmax() if n_classes > 1 else nn.Sigmoid()

    def forward(self, x, apply_activation=True):
        """
        Pushes the input through the network
        Args:
            x (tensor): batch of images with dim [B, C, H, W]
            apply_activation (bool): if True apply the Sigmoid on the output
        """
        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.activation(logits) if apply_activation else logits
