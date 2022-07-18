from .trainer_torch import TorchTrainer
from utils import *
from losses.precision_recall_f1 import *

import math
import torch
from torch import optim

# based on https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B1/segformer.b1.1024x1024.city.160k.py
class SegFormerTrainer(TorchTrainer):
    """
    Trainer for the SegFormer model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None,
                 loss_function_hyperparams=None, evaluation_interval=None, num_samples_to_visualize=None,
                 checkpoint_interval=None, load_checkpoint_path=None, segmentation_threshold=None,
                 use_channelwise_norm=False, blobs_removal_threshold=0, hyper_seg_threshold=False):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 4

        if num_epochs is None:
            num_epochs = 5000

        if optimizer_or_lr is None:
            default_lr = 0.00006
            optimizer_or_lr = SegFormerTrainer.get_default_optimizer_with_lr(default_lr, model.backbone)
            self.optimizer_or_lr_head = SegFormerTrainer.get_default_optimizer_with_lr(default_lr * 10, model.head)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = SegFormerTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model.backbone)
            self.optimizer_or_lr_head = SegFormerTrainer.get_default_optimizer_with_lr(optimizer_or_lr * 10, model.head)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_or_lr, step_size=1, gamma=1)

        if loss_function is None:
            self.ce_loss = torch.nn.CrossEntropyLoss()
            loss_function = lambda input, target: self.ce_loss(input, target.long())

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs, num_samples_to_visualize)

        preprocessing = None
        if use_channelwise_norm and dataloader.dataset in DATASET_STATS:
            def channelwise_preprocessing(x, is_gt):
                if is_gt:
                    return x[:1, :, :].float() / 255
                stats = DATASET_STATS[dataloader.dataset]
                x = x[:3, :, :].float()
                x[0] = (x[0] - stats['pixel_mean_0']) / stats['pixel_std_0']
                x[1] = (x[1] - stats['pixel_mean_1']) / stats['pixel_std_1']
                x[2] = (x[2] - stats['pixel_mean_2']) / stats['pixel_std_2']
                return x
            preprocessing = channelwise_preprocessing
        else:
            # convert samples to float32 \in [0, 1] & remove A channel;
            # convert ground truth to int \in {0, 1} & remove A channel
            preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        super().__init__(dataloader, model, preprocessing, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, scheduler, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold, use_channelwise_norm, blobs_removal_threshold, hyper_seg_threshold)
        
    def _train_step(self, model, device, train_loader, callback_handler):
        
        # use custom LR for head

        model.train()
        opt_backbone = self.optimizer_or_lr
        opt_head = self.optimizer_or_lr_head
        train_loss = 0
        for (x, y) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y = torch.squeeze(y, dim=1)  # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            preds = model(x, softmax=False)
            loss = self.loss_function(preds, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt_backbone.zero_grad()
            opt_head.zero_grad()
            loss.backward()
            opt_backbone.step()
            opt_head.step()
            callback_handler.on_train_batch_end()
            del x
            del y
        train_loss /= len(train_loader.dataset)
        callback_handler.on_epoch_end()
        self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
                y = torch.squeeze(y, dim=1)
                preds = model(x, softmax=False)
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
        return optim.Adam(model.parameters(), lr=lr)
