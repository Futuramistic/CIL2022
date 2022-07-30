"""
Adapted from https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction
The network we use is called "OurDinkNet50",

In turn, code based on:
"Codes of LinkNet based on https://github.com/snakers4/spacenet-three"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from functools import partial
from torch.autograd import Variable

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    """
    Module performing dilated convolutions followed by ReLu activations (4 times)
    """
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    """
    Decoding module used in the CRA-Net decoder (3 instances)
    Consists in Conv->BN->Relu->
                ConvTranspose->BN->Relu->
                Conv->BN->Relu
    """
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class PositionAttentionModule(nn.Module):
    """
    Module for applying spatial attention
    Some image coordinates can be more or less important according to its attention weight
    """
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class ChannelAttentionModule(nn.Module):
    """
    Module for applying per-channel attention
    Each channel can be more or less important according to its attention weight
    """
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out


class OurDinkNet50(nn.Module):
    """
    The CRA-Net
    @Note: The name is kept as-is from the original repo:
    https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction
    """
    def __init__(self, num_classes=1):
        super(OurDinkNet50, self).__init__()

        filters = [256, 512, 1024, 128]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self._first_conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding='same'),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(),
                                         nn.Conv2d(16, 32, kernel_size=3, padding='same'),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU()
                                         )
        self.first_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, kernel_size=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self._dblock32 = Dblock(32)
        self.dblock = Dblock(1024)
        self.pam3 = PositionAttentionModule(in_channels=1024)
        self.cam = ChannelAttentionModule()
        self.sample_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1))

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[3])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)  # original 32, 32
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1)

        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax()

        self.lam1 = Variable(torch.rand([]), requires_grad=True)
        self.lam2 = Variable(torch.rand([]), requires_grad=True)

    def forward(self, input, apply_activation=True):
        """
        Pushes the input through the network
        Args:
            input (tensor): batch of images with dim [B, C, H, W]
            apply_activation (bool): if True apply the Sigmoid on the output
        """
        # Encoder
        x = self.firstconv(input)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e3 = self.pam3(e3)
        e3 = self.dblock(e3)
        e4 = self.cam(e3)

        # Decoder
        d3 = self.decoder3(e4) + self.lam1 * e2
        d2 = self.decoder2(d3) + self.lam2 * e1
        x = self.first_conv(x)
        d1 = self.decoder1(d2+x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        res = self.finalconv3(out)
        refine = self.sample_conv(out)
        if apply_activation:
            res = self.activation(res)
        return refine, res
