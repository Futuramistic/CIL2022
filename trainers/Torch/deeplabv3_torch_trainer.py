import torch

from losses import MixedLoss
from losses.precision_recall_f1 import precision_recall_f1_score_torch
from trainers.trainer_torch import TorchTrainer
from utils import *
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DeepLabV3Trainer(TorchTrainer):
    """
    Trainer for the DeepLabV3 model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None,
                 loss_function_hyperparams=None, evaluation_interval=None, num_samples_to_visualize=None,
                 checkpoint_interval=None, load_checkpoint_path=None, segmentation_threshold=None,
                 use_channelwise_norm=False, blobs_removal_threshold=0, hyper_seg_threshold=False,
                 use_sample_weighting=False, use_adaboost=False, deep_adaboost=False,
                 f1_threshold_to_log_checkpoint=DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT):
        """
        Set omitted parameters to model-specific defaults, then call superclass __init__ function
        @Warning: some arguments depend on others not being None, so respect this order!

        Args:
            Refer to the TorchTrainer superclass for more details on the arguments
        """

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 4

        if num_epochs is None:
            num_epochs = 1

        if optimizer_or_lr is None:
            optimizer_or_lr = DeepLabV3Trainer.get_default_optimizer_with_lr(1e-4, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = DeepLabV3Trainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer_or_lr, mode="min", patience=15, factor=0.5)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_or_lr, step_size=1, gamma=1)

        if loss_function is None:
            loss_function = MixedLoss(10.0, 2.0)

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
                         segmentation_threshold, use_channelwise_norm, blobs_removal_threshold, hyper_seg_threshold,
                         use_sample_weighting, use_adaboost, deep_adaboost, f1_threshold_to_log_checkpoint)

    def _train_step(self, model, device, train_loader, callback_handler):
        """
        Train the model for 1 epoch

        @Note: WARNING: some models subclassing TorchTrainer overwrite this function, so make sure any changes here are
        reflected appropriately in these models' files

        Args:
            model: The model to train
            device: either 'cuda' or 'cpu'
            train_loader: train dataset loader object
            callback_handler: To be called when the train step is over

        Returns:
            train loss (float)
        """
        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0
        for (x, y, sample_idx) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            if x.shape[0] == 1:
                continue  # drop if the last batch has size of 1 (otherwise the deeplabv3 model crashes)
            preds = model(x, apply_activation=False)
            loss = self.loss_function(preds, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            if self.use_sample_weighting:
                threshold = getattr(self, 'last_hyper_threshold', self.segmentation_threshold)
                # weight based on F1 score of batch
                print("sample_idx: ", sample_idx)
                self.weights[sample_idx] =\
                    1.0 - precision_recall_f1_score_torch((preds.squeeze() >= threshold).float(), y)[-1].mean().item()
                print("weights :", self.weights)
            del x
            del y
        train_loss /= len(train_loader.dataset)
        f1_score = callback_handler.on_epoch_end()
        if f1_score is not None: # if None, we are in the first train step of a run with only a single batch
            self.scheduler.step(f1_score)
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
            for (x, y, _) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
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
                **({param: getattr(self.model, param)
                    for param in ['n_channels', 'n_classes', 'bilinear']
                    if hasattr(self.model, param)})}

    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        """
        Return the default optimizer for this network.
        Args:
            lr (float): Learning rate of the optimizer
            model: Model whose parameters we want to train
        """
        # return optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
        return optim.Adam(model.parameters(), lr=lr)
