"""
All losses take as input the groundtruth and the prediction tensors and output the loss 
value if not stated otherwise.
"""
import tensorflow as tf
import keras.backend as K


"""
Two implementations can be found online:
-> BCE + dice_loss
-> 0.5*BCE - dice_coeff
"""
def DiceBCELoss1(smooth=1e-6, logits=False):
    """
    BCE + dice_loss
    Args:
        smooth (float): Smoothing coefficient
        logits: whether the inputs are given as logits or not
    """
    def loss(targets, inputs):       
        inputs  = K.flatten(tf.cast(inputs, tf.float32))
        targets = K.flatten(tf.cast(targets, tf.float32))
        
        BCE = K.binary_crossentropy(targets, inputs, from_logits=logits)
        intersection = K.sum(targets*inputs)    
        dice_coeff = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
        dice_loss = 1 - dice_coeff
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    return loss


def DiceBCELoss2(smooth=1e-6, logits=False):
    """
    0.5*BCE - dice_coeff
    Args:
        smooth (float): Smoothing coefficient
        logits: whether the inputs are given as logits or not
    """
    def loss(targets, inputs):
        inputs  = K.flatten(tf.cast(inputs, tf.float32))
        targets = K.flatten(tf.cast(targets, tf.float32))
        
        BCE = K.binary_crossentropy(targets, inputs, from_logits=logits)
        intersection = K.sum(targets*inputs)    
        dice_coeff = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
        Dice_BCE = 0.5*BCE + dice_coeff
        return Dice_BCE

    return loss


def BCELoss(logits=False):
    """
    Wrapper for Binary Cross Entropy for other functions
    Args:
        logits: whether the inputs are given as logits or not
    """
    def loss(targets, inputs):
        inputs  = K.flatten(tf.cast(inputs, tf.float32))
        targets = K.flatten(tf.cast(targets, tf.float32))
        BCE = K.binary_crossentropy(targets, inputs, from_logits=logits)
        return BCE
    return loss
