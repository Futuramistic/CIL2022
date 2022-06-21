import tensorflow as tf
import keras.backend as K

def U2NET_BCE(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = K.BinaryCrossentropy(y_true, y_pred[0])
    loss1 = K.BinaryCrossentropy(y_true, y_pred[1])
    loss2 = K.BinaryCrossentropy(y_true, y_pred[2])
    loss3 = K.BinaryCrossentropy(y_true, y_pred[3])
    loss4 = K.BinaryCrossentropy(y_true, y_pred[4])
    loss5 = K.BinaryCrossentropy(y_true, y_pred[5])
    loss6 = K.BinaryCrossentropy(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6