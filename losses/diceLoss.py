import tensorflow as tf
import keras.backend as K


def DiceLossTF(smooth=1e-6):
    """
    Compute the Dice Loss
    Args:
        smooth (float): Smoothing coefficient
    """
    def loss(targets, inputs):
        inputs  = K.flatten(tf.cast(inputs, tf.float32))
        targets = K.flatten(tf.cast(targets, tf.float32))
        
        intersection = K.sum(targets*inputs)
        dice = (2.*intersection + smooth)/(K.sum(targets) + K.sum(inputs) + smooth)
        return 1 - dice
        
    return loss
