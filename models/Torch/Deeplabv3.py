"""
DeepLabV3 pretrained model
Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py for details
"""
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class Deeplabv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3(pretrained=True, progress=True)
        self.model.classifier = DeepLabHead(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, apply_activation=True):
        out = self.model(x)["out"]
        if apply_activation:
            out = self.sigmoid(out)
        return out
