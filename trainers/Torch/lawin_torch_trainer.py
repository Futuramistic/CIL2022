import torch
from torch import optim
from utils import *

from trainers.trainer_torch import TorchTrainer

class LawinTrainer(TorchTrainer):
    """
    Trainer for the SegFormer model.
    Based on :
    based on https://github.com/yan-hao-tian/lawin
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None,
                 loss_function_hyperparams=None, evaluation_interval=None, num_samples_to_visualize=None,
                 checkpoint_interval=None, load_checkpoint_path=None, segmentation_threshold=None,
                 use_channelwise_norm=True, blobs_removal_threshold=0, hyper_seg_threshold=False,
                 use_sample_weighting=False, f1_threshold_to_log_checkpoint=DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT):
        """
        Set omitted parameters to model-specific defaults, then call superclass __init__ function
        @Warning: some arguments depend on others not being None, so respect this order!

        Args:
            Refer to the TFTrainer superclass for more details on the arguments
        """

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 3

        if num_epochs is None:
            num_epochs = 5000

        default_lr = 0.00006
        head_lr_mult = 10

        if optimizer_or_lr is None:
            self.backbone_lr = default_lr
            self.head_lr = self.backbone_lr * head_lr_mult
            optimizer_or_lr = LawinTrainer.get_default_optimizer_with_lr(default_lr, model.backbone)
            self.optimizer_or_lr_head = LawinTrainer.get_default_optimizer_with_lr(self.head_lr, model.head)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            self.backbone_lr = optimizer_or_lr
            self.head_lr = self.backbone_lr * head_lr_mult
            optimizer_or_lr = LawinTrainer.get_default_optimizer_with_lr(self.backbone_lr, model.backbone)
            self.optimizer_or_lr_head = LawinTrainer.get_default_optimizer_with_lr(self.head_lr, model.head)
        else:
            self.backbone_lr = getattr(optimizer_or_lr, 'lr', None)
            self.head_lr = (self.backbone_lr if self.backbone_lr is not None else default_lr) * head_lr_mult
            self.optimizer_or_lr_head = LawinTrainer.get_default_optimizer_with_lr(self.head_lr, model.head)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_or_lr, step_size=1, gamma=1)

        if loss_function is None:
            self.ce_loss = torch.nn.CrossEntropyLoss()
            loss_function = lambda input, target: self.ce_loss(input, target.long())

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(batch_size)

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
                         segmentation_threshold, use_channelwise_norm, blobs_removal_threshold, False,  # last: hyper_seg_threshold
                         use_sample_weighting, f1_threshold_to_log_checkpoint)
        
    def _train_step(self, model, device, train_loader, callback_handler):
        """
        Train the model for one step. Uses Custom LR for head

        Args:
            model: The model to train
            device: either 'cuda' or 'cpu'
            train_loader: train dataset loader object
            callback_handler: To be called when the train step is over

        Returns:
            train loss (float)
        """

        model.train()
        opt_backbone = self.optimizer_or_lr
        opt_head = self.optimizer_or_lr_head
        train_loss = 0
        for (x, y, sample_idx) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y = torch.squeeze(y, dim=1)  # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            preds = model(x, apply_activation=False)
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
        """
        Evaluate the model. Called at the end of each epoch

        Args:
            model: model to evaluate
            device: either 'cuda' or 'cpu'
            test_loader: loader for the samples to evaluate the model on

        Returns:
            test loss (float)
        """
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y, sample_idx) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
                y = torch.squeeze(y, dim=1)
                preds = model(x, apply_activation=False)
                test_loss += self.loss_function(preds, y).item()
                del x
                del y
        test_loss /= len(test_loader.dataset)
        return test_loss

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        """
        return {**(super()._get_hyperparams()),
                **({f'opt_{param}': getattr(self, param)
                   for param in ['head_lr', 'backbone_lr']
                   if hasattr(self, param)}),
                **({param: getattr(self.model, param)
                   for param in ['align_corners', 'pretrained_backbone_path']
                   if hasattr(self.model, param)}),
                **({f'bb_{param}': getattr(self.model.backbone, param)
                   for param in ['num_classes', 'depths', 'img_size', 'patch_size', 'in_chans', 'embed_dims',
                                 'num_heads', 'mlp_ratios', 'qkv_bias', 'qk_scale', 'drop_rate', 'attn_drop_rate',
                                 'drop_path_rate', 'depths', 'sr_ratios']
                   if hasattr(self.model.backbone, param)}),
                **({f'head_{param}': getattr(self.model.head, param)
                   for param in ['in_channels', 'num_classes', 'feature_strides', 'dropout_rate', 'embedding_dim',
                                 'dropout_rate']
                   if hasattr(self.model.backbone, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        """
        Return the default optimizer for this network.
        Args:
            lr (float): Learning rate of the optimizer
            model: Model whose parameters we want to train
        """
        return optim.Adam(model.parameters(), lr=lr)
