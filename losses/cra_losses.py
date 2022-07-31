"""
Adapted from:

Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

All losses take as input the groundtruth and the prediction tensors and output the loss 
value if not stated otherwise.
"""

import torch.nn.functional as F
import torch
import torch.nn as nn

from utils import to_cuda
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
        self.criterion1 = to_cuda(dice_bce_loss_with_logits1())
        self.criterion2 = to_cuda(dice_bce_loss_with_logits())

    def __call__(self, labels, lower, outputs):
        lossr = self.criterion1(labels, lower)
        losss = self.criterion2(labels, outputs)
        loss = losss + 4 * lossr
        return loss


class cra_loss_with_vgg(nn.Module):
    """
    Alternative Loss for the CRA Net model, that includes an extra VGG loss and calculates
    a combined loss with the refined output
    """

    def __init__(self):
        super(cra_loss_with_vgg, self).__init__()
        self.criterion1 = to_cuda(dice_bce_loss_with_logits1())
        self.criterion2 = to_cuda(dice_bce_loss_with_logits())
        self.criterion3 = to_cuda(VGGPerceptualLoss())

    def __call__(self, labels, refined, outputs):
        lossr = self.criterion1(labels, refined)
        losss = self.criterion2(labels, outputs)
        lossv = self.criterion3(refined, labels)
        loss = losss + 2 * lossr + lossv
        return loss


class dice_bce_loss_with_logits(nn.Module):
    """
    Loss combining the BCE Loss and the Dice Loss
    Args:
        batch (bool): Whether batches or single samples are used
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
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=to_cuda(torch.Tensor([5.5])))


class dice_bce_loss_with_logits1(nn.Module):
    """
    Loss combining the BCE Loss and the Dice Loss (Version with different weighting)
    Args:
        batch (bool): Whether batches or single samples are used
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
        a = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=to_cuda(torch.Tensor([2.5])))
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Args:
            predict: A tensor of shape [N, *]
            target: A tensor of shape same with predict
        Raises:
            Exception: If unexpected reduction format appears
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        """
        Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        """
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


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
