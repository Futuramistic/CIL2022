from losses import *
import tensorflow as tf
import keras.backend as K

def DeepSupervisionLoss(function=DiceLoss,loss_weights=None,**kwargs):
    def loss(targets, inputs):
        targets = K.flatten(tf.cast(targets,tf.float32))
        inputs = tf.expand_dims(inputs, axis=-1)
        loss_sum = 0.0
        for i in range(0,len(loss_weights)):
            loss_sum += loss_weights[i]*function(**kwargs)(targets,K.flatten(tf.cast(inputs[i],tf.float32)))
        return loss_sum
    return loss