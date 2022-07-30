"""
All losses take as input the groundtruth and the prediction tensors and output the loss 
value if not stated otherwise.
"""
import tensorflow as tf
import keras.backend as K

from losses import FocalTverskyLoss


def IOUSeg(targets, inputs):
    """
    Compute the IOU Loss
    """
    inputs  = K.flatten(tf.cast(inputs, tf.float32))
    targets = K.flatten(tf.cast(targets, tf.float32))
    area_intersect = tf.reduce_sum(tf.multiply(targets, inputs))
    area_true = tf.reduce_sum(targets)
    area_pred = tf.reduce_sum(inputs)
    area_union = area_true + area_pred - area_intersect
    return 1-tf.math.divide_no_nan(area_intersect, area_union)


def HybridLoss(alpha=0.5, beta=0.5, gamma=1):
    """
    Compute a hybrid loss consisting of a Focal Tversky Loss and an IOU Loss
    Args:
        alpha (float): Refer to paper
        beta (float): Refer to paper
        gamma (float): Refer to paper
    """
    def loss(targets, inputs):
        loss_focal = FocalTverskyLoss(alpha=alpha,beta=beta,gamma=gamma)(targets,inputs)
        loss_iou = IOUSeg(targets, inputs)
        return loss_focal+loss_iou  # +loss_ssim
    return loss
