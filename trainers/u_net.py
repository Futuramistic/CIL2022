from .trainer_torch import TorchTrainer
from utils import *

import math
import torch
from torch import optim

# Some parameters taken from https://github.com/milesial/Pytorch-UNet/blob/master/train.py
class UNetTrainer(TorchTrainer):
    """
    Trainer for the U-Net model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer=None, scheduler=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 4  # 1

        if num_epochs is None:
            num_epochs = 1  # 5

        if optimizer is None:
            optimizer = optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        if loss_function is None:
            loss_function = torch.nn.CrossEntropyLoss()

        if evaluation_interval is None:
            evaluation_interval = 10

        # convert training samples to float32 \in [0, 1] & remove A channel;
        # convert test samples to int \in {0, 1} & remove A channel
        preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        super().__init__(dataloader, model, preprocessing, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer, scheduler, loss_function, evaluation_interval,
                         num_samples_to_visualize, checkpoint_interval)

    @staticmethod
    def get_default_optimizer_with_lr(lr):
        pass
