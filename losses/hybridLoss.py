from focalTversky import FocalTverskyLoss
import tensorflow as tf
import keras.backend as K

def IOUSeg(targets,inputs):
    inputs  =   K.flatten(tf.cast(inputs,tf.float32))
    targets =   K.flatten(tf.cast(targets,tf.float32))
    area_intersect = tf.reduce_sum(tf.multiply(targets, inputs))
    area_true = tf.reduce_sum(targets)
    area_pred = tf.reduce_sum(inputs)
    area_union = area_true + area_pred - area_intersect
    return 1-tf.math.divide_no_nan(area_intersect, area_union)

def HybridLoss(alpha=0.5,beta=0.5,gamma=1):
    def loss(targets, inputs):
        loss_focal = FocalTverskyLoss(alpha=alpha,beta=beta,gamma=gamma)(targets,inputs)
        loss_iou = IOUSeg(targets, inputs)
        return loss_focal+loss_iou#+loss_ssim
    return loss