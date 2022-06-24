import tensorflow as tf
import keras.backend as K
from losses import *

# Apply any loss to targets
def U2NET_loss(loss_func = BCELoss(logits=False)):

    def loss(targets, inputs):
        targets = K.flatten(tf.cast(targets,tf.float32))
        inputs = tf.expand_dims(inputs, axis=-1)
        loss0 = loss_func(targets, K.flatten(tf.cast(inputs[0],tf.float32)))
        loss1 = loss_func(targets, K.flatten(tf.cast(inputs[1],tf.float32)))
        loss2 = loss_func(targets, K.flatten(tf.cast(inputs[2],tf.float32)))
        loss3 = loss_func(targets, K.flatten(tf.cast(inputs[3],tf.float32)))
        loss4 = loss_func(targets, K.flatten(tf.cast(inputs[4],tf.float32)))
        loss5 = loss_func(targets, K.flatten(tf.cast(inputs[5],tf.float32)))
        loss6 = loss_func(targets, K.flatten(tf.cast(inputs[6],tf.float32)))
        return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    
    return loss