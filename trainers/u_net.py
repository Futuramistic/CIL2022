from .trainer_torch import TorchTrainer
from utils import *
from losses.precision_recall_f1 import *

import math
import torch
from torch import optim

# Some parameters taken from https://github.com/milesial/Pytorch-UNet/blob/master/train.py
class UNetTrainer(TorchTrainer):
    """
    Trainer for the U-Net model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None, load_checkpoint_path=None,
                 segmentation_threshold=None):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 4  # 1

        if num_epochs is None:
            num_epochs = 1  # 5

        if optimizer_or_lr is None:
            optimizer_or_lr = UNetTrainer.get_default_optimizer_with_lr(1e-5, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = UNetTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_or_lr, step_size=1, gamma=1)

        if loss_function is None:
            loss_function = torch.nn.BCELoss()

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs, num_samples_to_visualize)

        # convert samples to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in {0, 1} & remove A channel
        preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        super().__init__(dataloader, model, preprocessing, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, scheduler, loss_function, evaluation_interval,
                         num_samples_to_visualize, checkpoint_interval, load_checkpoint_path, segmentation_threshold)
        
    def _train_step(self, model, device, train_loader, callback_handler):
        # unet y may not be squeezed like in torch trainer, dtype is float for BCE
        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0
        for (x, y) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            preds = model(x)
            loss = self.loss_function(preds, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            del x
            del y
        train_loss /= len(train_loader.dataset)
        callback_handler.on_epoch_end()
        self.scheduler.step()
        return train_loss
    
    def _eval_step(self, model, device, test_loader):
        # unet y may not be squeezed like in torch trainer, dtype is float for BCE
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
                preds = model(x)
                test_loss += self.loss_function(preds, y).item()
                del x
                del y
        test_loss /= len(test_loader.dataset)
        return test_loss

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self.model, param)
                   for param in ['n_channels', 'n_classes', 'bilinear']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        return optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
