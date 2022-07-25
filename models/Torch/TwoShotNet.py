import torch
import torch.nn as nn
from models.Torch.Unet import UNet


class TwoShotNet(nn.Module):
    """
    @Note: Custom Network

    A Network formed by the concatenation of 2 UNets.
    The idea is the following: First feed the images through the first UNet, get a pre-segmentation, then concatenate
    this segmentation with original input and feed it through the second UNet.

    This architecture did not yield any improvement over the UNet baseline
    """
    def __init__(self):
        super(TwoShotNet, self).__init__()
        self.net = UNet(3+1, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        B, C, H, W = x.shape
        zero = torch.zeros((B, 1, H, W), device=self.device)
        x1 = torch.cat((zero, x), 1)
        o1 = self.net(x1)
        o1 = (o1 > 0.8).float()
        x2 = torch.cat((o1, x), 1)
        o2 = self.net(x2)
        o = torch.max(o1, o2)
        return o
