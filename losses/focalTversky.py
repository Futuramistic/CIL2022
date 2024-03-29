"""
All losses take as input the groundtruth and the prediction tensors and output the loss 
value if not stated otherwise.
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def FocalTverskyLoss(alpha=.3, beta=.7, gamma=.75, smooth=1e-6):
    """
    Focal Tversky Loss implentation. Refer to Paper: https://arxiv.org/pdf/1706.05721.pdf for more information
    Args:
        alpha (float): Refer to paper
        beta (float): Refer to paper
        gamma (float): Refer to paper
        smooth (float): Smoothing coefficient
    """

    def loss(targets, inputs):
        # Flatten label and prediction tensors
        inputs  = K.flatten(tf.cast(inputs,tf.float32))
        targets = K.flatten(tf.cast(targets,tf.float32))

        # True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = K.pow((1 - Tversky), gamma)

        return FocalTversky

    return loss