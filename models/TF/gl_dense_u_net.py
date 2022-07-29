"""
code re-written based on original TF-slim implementation from:
https://github.com/cugxyy/GL-Dense-U-Net/blob/master/Model/GL_Dense_U_Net.py
"""

import tensorflow as tf
import tensorflow.keras as K

from utils import DEFAULT_TF_INPUT_SHAPE


def GLDenseUNet(input_shape=(None, 1, 3),
                growth_rate=16,
                layers_per_block=(4, 5, 7, 10, 12),
                conv2d_activation='relu',
                num_classes=2,
                input_resize_dim=256,
                l2_regularization_param=1e-5):
    """
    GLDenseUNet model.

    @Note:
    This is a function and not a tf.keras.Model subclass because we're using the Keras functional API here to
    construct the model's non-linear topology, and the functional API is not compatible with subclassing
    tf.keras.Model. Yet, we want to maintain the illusion of "GLDenseUNet" returning a tf.keras.Model.
    """

    if isinstance(layers_per_block, str):
        layers_per_block = list(eval(layers_per_block))
    num_blocks = len(layers_per_block)

    def l2_reg():
        return K.regularizers.l2(l2_regularization_param)

    def dense_block(x, block_idx):
        dense_block_out = []
        for conv_layer_idx in range(layers_per_block[block_idx]):
            conv_block = K.Sequential()
            conv_block.add(K.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True))
            conv_block.add(K.layers.ReLU())
            conv_block.add(K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                           activation=conv2d_activation, kernel_regularizer=l2_reg()))
            conv_block_out = conv_block(x)
            dense_block_out.append(conv_block_out)
            x = tf.concat([conv_block_out, x], axis=3)
        x = tf.concat(dense_block_out, axis=3)
        return x

    def lau(x):
        conv2d_args = {'activation': conv2d_activation,
                       'padding': 'SAME',
                       'kernel_regularizer': l2_reg()}
        conv2d_t_args = {'activation': conv2d_activation,
                         'padding': 'SAME'}
        x_shape = x.shape.as_list()

        conv_1x1 = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(1, 1), strides=(1, 1), **conv2d_args)(x)
        conv_7x7 = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(7, 7), strides=(2, 2), **conv2d_args)(x)
        conv_5x5 = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(5, 5), strides=(4, 4), **conv2d_args)(x)
        conv_3x3 = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(3, 3), strides=(8, 8), **conv2d_args)(x)

        conv_3x3_up = K.layers.Conv2DTranspose(filters=x_shape[-1], kernel_size=(3, 3), strides=(2, 2),
                                               **conv2d_t_args)(conv_3x3)
        concat_3x5 = tf.concat([conv_3x3_up, conv_5x5], axis=3)
        concat_3x5_up = K.layers.Conv2DTranspose(filters=x_shape[-1], kernel_size=(3, 3), strides=(2, 2),
                                                 **conv2d_t_args)(concat_3x5)
        concat_5x7 = tf.concat([concat_3x5_up, conv_7x7], axis=3)
        concat_5x7_up = K.layers.Conv2DTranspose(filters=x_shape[-1], kernel_size=(3, 3), strides=(2, 2),
                                                 **conv2d_t_args)(concat_5x7)
        multi_1x7 = conv_1x1 * concat_5x7_up
        final_conv_1x1 = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(1, 1), strides=(1, 1),
                                         **conv2d_args)(multi_1x7)
        return final_conv_1x1

    def gau(x):
        conv2d_args = {'activation': conv2d_activation,
                       'padding': 'SAME',
                       'kernel_regularizer': l2_reg()}
        conv2d_t_args = {'activation': conv2d_activation,
                         'padding': 'SAME'}
        x_shape = x.shape.as_list()

        global_pool = K.layers.AvgPool2D(pool_size=(x_shape[2], x_shape[2]), strides=(1, 1), padding='SAME')(x)
        global_pool = K.layers.Conv2D(filters=x_shape[-1], kernel_size=(1, 1), strides=(1, 1),
                                      **conv2d_args)(global_pool)
        global_pool = K.layers.Conv2DTranspose(filters=x_shape[-1], kernel_size=(3, 3), strides=(2, 2),
                                               **conv2d_t_args)(global_pool)
        return global_pool

    def __build_model(inputs):
        conv2d_args = {'activation': conv2d_activation,
                       'kernel_regularizer': l2_reg()}
        conv2d_t_args = {'activation': conv2d_activation}
        concats = []

        if input_resize_dim is not None:
            x = K.layers.Resizing(input_resize_dim, input_resize_dim, interpolation='bilinear')(inputs)
        else:
            x = inputs
        x = K.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **conv2d_args)(x)
        for block_idx in range(num_blocks):
            dense_block_out = dense_block(x, block_idx)
            x = tf.concat([x, dense_block_out], axis=3)
            concats.append(x)
            if block_idx < num_blocks - 1:
                x = K.layers.Conv2D(filters=x.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                    **conv2d_args)(x)
                x = K.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='SAME')(x)

        x_last = lau(x)
        for rising_block_idx, falling_block_idx in enumerate(range(num_blocks - 1, 0, -1)):
            num_x_last_channels = x_last.get_shape()[-1]
            x = K.layers.Conv2DTranspose(filters=x_last.shape[-1], kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                         **conv2d_t_args)(x_last)
            gau_out = gau(x_last)
            x = tf.concat([x, gau_out], axis=3)
            gau_concat = concats[len(concats) - rising_block_idx - 2]
            if rising_block_idx != 1:
                gau_concat = lau(gau_concat)
            x = tf.concat([x, gau_concat], axis=3)
            x = K.layers.Conv2D(filters=num_x_last_channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                activation=None, kernel_regularizer=None)(x)
            x = dense_block(x, falling_block_idx)
            x_last = x

        if input_resize_dim is not None:
            x = K.layers.Resizing(inputs.shape[1], inputs.shape[2], interpolation='bilinear')(x)
            
        x = K.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                            activation='softmax', kernel_regularizer=None)(x)
        return x

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name='GLDenseUNet')
    
    # store hyperparameters so GLDenseUNetTrainer._get_hyperparams finds them
    model.growth_rate = growth_rate
    model.layers_per_block = layers_per_block
    model.conv2d_activation = conv2d_activation
    model.num_classes = num_classes
    model.input_resize_dim = input_resize_dim
    model.l2_regularization_param = l2_regularization_param
    
    return model
