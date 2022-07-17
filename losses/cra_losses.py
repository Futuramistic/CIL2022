"""
Adapted from:

Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

import torch.nn.functional as F
import torch
import torch.nn as nn

from losses.vgg_loss import VGGPerceptualLoss

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class cra_loss(nn.Module):
    """
    Loss used by the CRA Net model
    """

    def __init__(self):
        super(cra_loss, self).__init__()
        self.criterion1 = dice_bce_loss_with_logits1().cuda()
        self.criterion2 = dice_bce_loss_with_logits().cuda()

    def __call__(self, labels, lower, outputs):
        lossr = self.criterion1(labels, lower)
        losss = self.criterion2(labels, outputs)
        loss = losss + 4 * lossr
        return loss


class cra_loss_with_vgg(nn.Module):
    """
    Alternative Loss for the CRA Net model, that includes an extra VGG loss
    """

    def __init__(self):
        super(cra_loss_with_vgg, self).__init__()
        self.criterion1 = dice_bce_loss_with_logits1().cuda()
        self.criterion2 = dice_bce_loss_with_logits().cuda()
        self.criterion3 = VGGPerceptualLoss().cuda()

    def __call__(self, labels, refined, outputs):
        lossr = self.criterion1(labels, refined)
        losss = self.criterion2(labels, outputs)
        lossv = self.criterion3(refined, labels)
        loss = losss + 2 * lossr + lossv
        return loss


class dice_bce_loss_with_logits(nn.Module):
    """
    Loss combining the BCE Loss and the Dice Loss
    """

    def __init__(self, batch=True):
        super(dice_bce_loss_with_logits, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([5.5]).cuda())


class dice_bce_loss_with_logits1(nn.Module):
    """
    Loss combining the BCE Loss and the Dice Loss (Version with different weighting)
    """

    def __init__(self, batch=True):
        super(dice_bce_loss_with_logits1, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=torch.Tensor([2.5]).cuda())
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


# --------------------------- HELPER FUNCTIONS ---------------------------

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    Args:
        l (Iterable): Collection to apply the mean upon
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
