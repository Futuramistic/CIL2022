from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import numpy as np


class DeepLabV3PlusGAN(nn.Module):
    """
    @Note: Custom Network

    Adaptation of the original DeepLabV3 model with an additional Discriminator for GAN Training.
    The idea is to train this hybrid model both with the original losses and with a GAN Loss retrieved from
    a minimax game against a Discriminator that discriminates between real segmentations and generated ones.

    The performed tests did not seem to given improvements and made training more unstable.
    """
    def __init__(self, cnn_discriminator=True):
        super().__init__()
        self.model = deeplabv3(pretrained=True, progress=True)
        self.model.classifier = DeepLabHead(2048, 1)  # Define the head
        # Define either a CNN-based or a a Fully Connected discriminator network
        self.discriminator = CNN_Discriminator() if cnn_discriminator else FC_Discriminator()
        # Count the number of parameters
        nb_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        nb_model_classifier_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
        nb_discriminator_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f'# MODEL PARAMS: {nb_model_params}')
        print(f'# MODEL CLASSIFIER PARAMS: {nb_model_classifier_params}')
        print(f'# DISCRIMINATOR PARAMS: {nb_discriminator_params}')

    def get_discriminator(self):
        return self.discriminator

    def forward(self, x, apply_activation=True):
        """
        Pushes the input through the network
        Args:
            x (tensor): batch of images with dim [B, C, H, W]
            apply_activation (bool): if True apply the Sigmoid on the output
        """
        out = self.model(x)["out"]
        if apply_activation:
            out = self.sigmoid(out)
        return out


class CNN_Discriminator(nn.Module):
    """
    CNN-based discriminator
    Low parameter overhead (~17k)
    """
    def __init__(self):
        super(CNN_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """
            Args:
                in_filters (int): Number of input filters
                out_filters (int): Number of output filters
                bn (bool): Whether to apply Batch Normalization
            """
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        channels = 1
        img_size = 400

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class FC_Discriminator(nn.Module):
    """
    Fully Connected discriminator
    Heavily parametrized
    """
    def __init__(self):
        super(FC_Discriminator, self).__init__()
        img_shape = (400, 400)
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
