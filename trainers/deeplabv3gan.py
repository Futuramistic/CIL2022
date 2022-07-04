from .trainer_torch import TorchTrainer
from utils import *

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.autograd import Variable
# from torch import Tensor

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class DeepLabV3PlusGANTrainer(TorchTrainer):
    """
    Trainer for the dual DeepLabV3 + GAN model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, scheduler=None, loss_function=None,
                 loss_function_hyperparams=None, evaluation_interval=None, num_samples_to_visualize=None,
                 checkpoint_interval=None, load_checkpoint_path=None, segmentation_threshold=None,
                 adv_lambda=0.1, adv_lr=1e-4):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        self.adv_lambda = adv_lambda
        self.adv_lr = adv_lr

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 4

        if num_epochs is None:
            num_epochs = 1

        if optimizer_or_lr is None:
            optimizer_or_lr = DeepLabV3PlusGANTrainer.get_default_optimizer_with_lr(1e-5, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = DeepLabV3PlusGANTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        self.D = model.get_discriminator()
        # Discriminator optimizer
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.adv_lr, betas=(0.9, 0.999))

        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer_or_lr, mode="min", patience=15, factor=0.5)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_or_lr, step_size=1, gamma=1)

        if loss_function is None:
            loss_function = MixedLoss(10.0, 2.0)

        # Adversarial Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs,
                                                                             num_samples_to_visualize)

        mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
        normalizer = transforms.Normalize(mean, std)
        # convert samples to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in {0, 1} & remove A channel

        def preprocessing(x, is_gt):
            res = (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)
            if not is_gt:
                res = normalizer(res)
            return res

        super().__init__(dataloader, model, preprocessing, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, scheduler, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold)

    def _train_step(self, model, device, train_loader, callback_handler):

        model.train()
        self.D.train()

        opt = self.optimizer_or_lr
        train_loss = 0
        for (x, y) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)

            # Adversarial ground truths
            valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(y.type(Tensor))  # original uses x, but we probably need y for our purposes

            # ================
            # Train generator

            if x.shape[0] == 1:
                continue  # drop if the last batch has size of 1 (otherwise the deeplabv3 model crashes)
            gen_imgs = model(x)
            loss = self.loss_function(gen_imgs, y)
            loss += self.adv_lambda * self.adversarial_loss(self.D(gen_imgs), valid)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # ====================
            # Train Discriminator

            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(self.D(real_imgs), valid)
            fake_loss = self.adversarial_loss(self.D(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            callback_handler.on_train_batch_end()
            del x
            del y

        train_loss /= len(train_loader.dataset)
        f1_score = callback_handler.on_epoch_end()
        self.scheduler.step(f1_score)
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        self.D.eval()

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
        # return optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
        return optim.Adam(model.parameters(), lr=lr)


# Losses used by this model
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
