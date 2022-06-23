from .trainer_torch import TorchTrainer
from utils import *

import torch
from torch import optim
from torch.autograd import Variable
from losses.cra_losses import dice_bce_loss


class U2NetTrainer(TorchTrainer):
    """
    Trainer for the CRA-Net model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None,
                 loss_function_hyperparams=None, evaluation_interval=None, num_samples_to_visualize=None,
                 checkpoint_interval=None, load_checkpoint_path=None, segmentation_threshold=None):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 8

        train_set_size, test_set_size, unlabeled_test_set_size = dataloader.get_dataset_sizes(split=split)
        steps_per_training_epoch = train_set_size // batch_size

        if num_epochs is None:
            num_epochs = math.ceil(100000 / steps_per_training_epoch)

        if num_epochs is None:
            num_epochs = 1

        if optimizer_or_lr is None:
            optimizer_or_lr = U2NetTrainer.get_default_optimizer_with_lr(1e-3, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = U2NetTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if loss_function is None:
            loss_function = dice_bce_loss()

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs,
                                                                             num_samples_to_visualize)

        # convert samples to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in {0, 1} & remove A channel
        preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        super().__init__(dataloader, model, preprocessing, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, scheduler, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold)

    def _train_step(self, model, device, train_loader, callback_handler):

        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0
        for (inputs, labels) in train_loader:
            labels = torch.squeeze(labels, dim=1)
            # TODO: May want to uncomment .cuda() if not available
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.float().cuda()
            opt.zero_grad()
            outputs, lower = model.forward(inputs)
            outputs = torch.squeeze(outputs, dim=1)
            lower = torch.squeeze(lower, dim=1)
            loss = self.loss_function(labels, lower, outputs)

            train_loss += loss.item()

            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()

        train_loss /= len(train_loader.dataset)
        callback_handler.on_epoch_end()
        self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (inputs, labels) in test_loader:
                labels = torch.squeeze(labels, dim=0)
                # TODO: May want to uncomment .cuda() if not available
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                labels = labels.float().cuda()

                outputs, lower = model.forward(inputs)
                outputs = torch.squeeze(outputs, dim=1)
                lower = torch.squeeze(lower, dim=1)
                loss = self.loss_function(labels, lower, outputs)

                test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        return test_loss

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self.model, param)
                    for param in ['n_channels', 'n_classes', 'bilinear']
                    if hasattr(self.model, param)})}

    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        return optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)