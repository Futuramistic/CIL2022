"""
All losses take as input the groundtruth and the prediction tensors and output the loss 
value if not stated otherwise.
"""
import tensorflow as tf
import keras.backend as K

from losses import DiceLossTF


def DeepSupervisionLoss(function=DiceLossTF, loss_weights=None, **kwargs):
    """
    Wrapper used for the deep supervision loss that sums weighted loss of each desired output
    """
    def loss(targets, inputs):
        targets = K.flatten(tf.cast(targets, tf.float32))
        inputs = tf.expand_dims(inputs, axis=-1)
        loss_sum = 0.0
        for i in range(0, len(loss_weights)):
            loss_sum += loss_weights[i]*function(**kwargs)(targets, K.flatten(tf.cast(inputs[i], tf.float32)))
        return loss_sum
    return loss
