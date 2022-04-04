import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import *

def conv2d(inputs, filters=64, padding='same', kernel_init='he_normal', dropout=0.5, normalize=False):
    convo = Conv2D(filters=filters, kernel_size=3, padding=padding, kernel_initializer=kernel_init)(inputs)
    if normalize:
        convo = BatchNormalization()(convo)
    convo = Activation(activation='relu')(convo)
    convo = Dropout(dropout)(convo)
    convo = Conv2D(filters=filters, kernel_size=3, padding=padding, kernel_initializer=kernel_init)(convo)
    if normalize:
        convo = BatchNormalization()(convo)
    convo = Activation(activation='relu')(convo)
    convo = Dropout(dropout)(convo)
    return convo


def down_batch(inputs, filters=64, padding='same', kernel_init='he_normal', dropout=0.5,normalize=False):
    convo1 = conv2d(inputs, filters=filters, padding=padding, kernel_init=kernel_init, dropout=dropout, normalize=normalize)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(convo1)
    return (convo1, pool1)


def upConovo(inputs, filters, padding, kernel_init, dropout,normalize):
    up = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding=padding,
                         kernel_initializer=kernel_init)(inputs)
    if normalize:
        up = BatchNormalization()(up)
    return up


def up_batch(inputs, merger, filters=64, padding='same', kernel_init='he_normal', dropout=0.5,normalize=False):
    upConvo = upConovo(inputs, filters, padding, kernel_init, dropout,normalize)
    merge = Concatenate(axis=3)([merger, upConvo])
    convo = conv2d(merge, filters=filters, padding=padding, kernel_init=kernel_init, dropout=dropout,normalize=normalize)
    return convo


# Standard U-Net implementation
def UNetTF(input_size, dropout=0.0, padding='same', kernel_init='he_normal',
         IOU_threshold=0.6, lr=1e-4, name=None, normalize=False,additional_params=None):
    nb_filters = [32, 64, 128, 256, 512]
    inputs = Input(input_size)

    # GOING DOWN
    (convo1, pool1) = down_batch(inputs, nb_filters[0], padding, kernel_init, dropout,normalize)
    (convo2, pool2) = down_batch(pool1, nb_filters[1], padding, kernel_init, dropout,normalize)
    (convo3, pool3) = down_batch(pool2, nb_filters[2], padding, kernel_init, dropout,normalize)
    (convo4, pool4) = down_batch(pool3, nb_filters[3], padding, kernel_init, dropout,normalize)

    # BOTTOM
    convo5 = conv2d(pool4, nb_filters[4], padding, kernel_init, dropout,normalize)

    # GOING UP
    convo6 = up_batch(convo5, convo4, nb_filters[3], padding, kernel_init, dropout,normalize)
    convo7 = up_batch(convo6, convo3, nb_filters[2], padding, kernel_init, dropout,normalize)
    convo8 = up_batch(convo7, convo2, nb_filters[1], padding, kernel_init, dropout,normalize)
    convo9 = up_batch(convo8, convo1, nb_filters[0], padding, kernel_init, dropout,normalize)

    output = Conv2D(filters=1, kernel_size=(1, 1), padding=padding, activation='sigmoid',
                    kernel_initializer=kernel_init)(convo9)

    model = tf.keras.Model(inputs=inputs, outputs=output, name=name)

    return model