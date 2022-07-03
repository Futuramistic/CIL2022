from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import numpy as np


class DeepLabV3PlusGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3(pretrained=True, progress=True)
        self.model.classifier = DeepLabHead(2048, 1)
        self.discriminator = Discriminator()

    def get_discriminator(self):
        return self.discriminator

    def forward(self, x):
        o1 = self.model(x)["out"]
        return o1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        img_shape = (400, 400)  # TODO parametrize this
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity