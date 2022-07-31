import warnings
import torch
import utils

from torch.utils.data import DataLoader as torchDL, Dataset, Subset
from .torchDataset import SegmentationDataset
from .dataloader import DataLoader
from .dataloader_torch import TorchDataLoader
from torchvision import transforms
from models import *


class TorchDataLoaderRLSupervised(TorchDataLoader):
    def __init__(self, dataset="original", use_geometric_augmentation=False, use_color_augmentation=False,
                 aug_contrast=[0.8,1.2], aug_brightness=[0.8, 1.2], aug_saturation=[0.8,1.2], use_adaboost=False):
        super().__init__(dataset, use_geometric_augmentation=use_geometric_augmentation,
                         use_color_augmentation=use_color_augmentation, aug_contrast=aug_contrast,
                         aug_brightness=aug_brightness, aug_saturation=aug_saturation, use_adaboost=use_adaboost,
                         use_rl_supervision=True)
