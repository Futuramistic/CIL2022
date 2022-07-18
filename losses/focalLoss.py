import tensorflow as tf
import keras.backend as K


def FocalLoss(alpha=.25, gamma=2., logits=False):
    """
    Focal Loss implentation. Refer to Paper: https://arxiv.org/pdf/1708.02002.pdf for more information
    Args:
        alpha (float): Refer to paper
        gamma (float): Refer to paper
        logits: whether the inputs are given as logits or not
    """
    def loss(targets, inputs):
        inputs  = K.flatten(tf.cast(inputs, tf.float32))
        targets = K.flatten(tf.cast(targets, tf.float32))
        
        BCE = K.binary_crossentropy(targets, inputs, from_logits=logits)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        
        return focal_loss
        
    return loss