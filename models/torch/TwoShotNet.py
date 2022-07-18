import torch
import torch.nn as nn
from models.torch.Unet import UNet


class TwoShotNet(nn.Module):
    def __init__(self):
        super(TwoShotNet, self).__init__()
        self.net = UNet(3+1, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        # assert len(x.shape) == 4
        B, C, H, W = x.shape
        # assert C == 3
        zero = torch.zeros((B, 1, H, W), device=self.device)
        x1 = torch.cat((zero, x), 1)
        o1 = self.net(x1)
        o1 = (o1 > 0.8).float()
        # assert len(o1.shape) == 4
        # print(':::::::::::::', o1.shape)
        x2 = torch.cat((o1, x), 1)
        o2 = self.net(x2)
        o = torch.max(o1, o2)
        return o
