"""The U-Net Exp model implemented in Tensorflow as well as its sub-components
"""
import tensorflow.keras as K

from keras.layers import *
from tensorflow.keras.layers import *
from .blocks import *
from utils import *


def UNetTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
           name="UNetTF",
           dropout=0.5,
           kernel_init='he_normal',
           normalize=True,
           up_transpose=True,
           kernel_regularizer=K.regularizers.l2(),
           use_learnable_pool=False,
           **kwargs):
    #K.mixed_precision.set_global_policy('mixed_float16')
    def __build_model(inputs):
        nb_filters = [32, 64, 128, 256, 512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        down_args = {
            'dropout': dropout,
            'kernel_init': kernel_init,
            'normalize': normalize,
            'kernel_regularizer': kernel_regularizer
        }

        up_args = {
            'dropout': dropout,
            'kernel_init': kernel_init,
            'normalize': normalize,
            'up_convo': up_block,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'filters': 1,
            'kernel_size': (3, 3),
            'padding': 'same',
            'kernel_initializer': kernel_init,
            'kernel_regularizer': kernel_regularizer,
        }

        convo1, pool1 = Down_Block(name=name + "-down-block-1", filters=nb_filters[0], kernel_size=5, **down_args)(inputs)
        convo2, pool2 = Down_Block(name=name + "-down-block-2", filters=nb_filters[1], kernel_size=5, **down_args)(pool1)
        convo3, pool3 = Down_Block(name=name + "-down-block-3", filters=nb_filters[2], **down_args)(pool2)
        convo4, pool4 = Down_Block(name=name + "-down-block-4", filters=nb_filters[3], **down_args)(pool3)

        convo5 = Convo_Block(name=name + "-convo-block", filters=nb_filters[4], dilation_rate=2, **down_args)(pool4)

        up1 = Up_Block(name=name + "-up-block-1", filters=nb_filters[3], **up_args)(x=convo5, merger=[convo4])
        up2 = Up_Block(name=name + "-up-block-2", filters=nb_filters[2], **up_args)(x=up1, merger=[convo3])
        up3 = Up_Block(name=name + "-up-block-3", filters=nb_filters[1], **up_args)(x=up2, merger=[convo2])
        up4 = Up_Block(name=name + "-up-block-4", filters=nb_filters[0], **up_args)(x=up3, merger=[convo1])
        out = Conv2D(name=name + "-final-convo", **out_args)(up4)
        sigmoid = Activation(tf.nn.sigmoid, dtype=tf.float32)
        return sigmoid(out)

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.up_transpose = up_transpose
    model.kernel_regularizer = kernel_regularizer
    return model


def UNet3PlusTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
                name="UNet3PlusTF",
                dropout=0.5,
                kernel_init='he_normal',
                normalize=True,
                kernel_regularizer=K.regularizers.l2(),
                use_learnable_pool=False,
                deep_supervision=False,
                cgm=False,
                cgm_dropout=0.1,
                **kwargs):
    #K.mixed_precision.set_global_policy('mixed_float16')
    def __build_model(inputs):
        nb_filters = [32, 64, 128, 256, 512, 64, 320]

        down_args = {
            'dropout': dropout,
            'kernel_init': kernel_init,
            'normalize': normalize,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'filters': 1,
            'kernel_size': (3, 3),
            'padding': 'same',
            'kernel_initializer': kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        convo_trans_args = {
            'kernel_size': (4, 4),
            'strides': (2, 2),
            'padding': 'same',
            'kernel_initializer': kernel_init,
            'kernel_regularizer': kernel_regularizer
        }
        inputs = K.applications.vgg19.preprocess_input(tf.cast(inputs,dtype=tf.float32))
        convo1, pool1 = Down_Block(name=name + "-down-block-1", filters=nb_filters[0], **down_args)(inputs)
        convo2, pool2 = Down_Block(name=name + "-down-block-2", filters=nb_filters[1], **down_args)(pool1)
        convo3, pool3 = Down_Block(name=name + "-down-block-3", filters=nb_filters[2], **down_args)(pool2)
        convo4, pool4 = Down_Block(name=name + "-down-block-4", filters=nb_filters[3], **down_args)(pool3)

        convo5 = Convo_Block(name=name + "-convo-block", filters=nb_filters[4], **down_args)(pool4)

        convo3_4 = MaxPool2D((2, 2), 2, 'same')(convo3)
        convo3_4 = Convo_Block(name=name + "-convo3_4", filters=nb_filters[5], **down_args)(convo3_4)

        convo2_4 = MaxPool2D((4, 4), 4, 'same')(convo2)
        convo2_4 = Convo_Block(name=name + "-convo2_4", filters=nb_filters[5], **down_args)(convo2_4)

        convo1_4 = MaxPool2D((8, 8), 8, 'same')(convo1)
        convo1_4 = Convo_Block(name=name + "-convo1_4", filters=nb_filters[5], **down_args)(convo1_4)

        convo4_4 = Convo_Block(name=name + "-convo4_4", filters=nb_filters[5], **down_args)(convo4)

        up_convo5 = Conv2DTranspose(name=name + "-up_convo5", filters=nb_filters[5], **convo_trans_args)(convo5)
        up1 = Concatenate(axis=3)([up_convo5, convo4_4, convo3_4, convo2_4, convo1_4])
        up1 = Convo_Block(name=name + "-up-1", filters=nb_filters[6], **down_args)(up1)

        convo5_3 = UpSampling2D(name=name + "-up5_3", size=(4, 4), interpolation='bilinear')(convo5)
        convo5_3 = Convo_Block(name=name + "-convo5_3", filters=nb_filters[5], **down_args)(convo5_3)

        convo2_3 = MaxPool2D((2, 2), 2, 'same')(convo2)
        convo2_3 = Convo_Block(name=name + "-convo2_3", filters=nb_filters[5], **down_args)(convo2_3)

        convo1_3 = MaxPool2D((4, 4), 4, 'same')(convo1)
        convo1_3 = Convo_Block(name=name + "-convo1_3", filters=nb_filters[5], **down_args)(convo1_3)

        convo3_3 = Convo_Block(name=name + "-convo3_3", filters=nb_filters[5], **down_args)(convo3)

        up_convo4 = Conv2DTranspose(name=name + "-up_convo4", filters=nb_filters[5], **convo_trans_args)(up1)
        up2 = Concatenate(axis=3)([up_convo4, convo3_3, convo2_3, convo1_3, convo5_3])
        up2 = Convo_Block(name=name + "-up-2", filters=nb_filters[6], **down_args)(up2)

        convo5_2 = UpSampling2D(name=name + "-up5_2", size=(8, 8), interpolation='bilinear')(convo5)
        convo5_2 = Convo_Block(name=name + "-convo5_2", filters=nb_filters[5], **down_args)(convo5_2)

        convo4_2 = UpSampling2D(name=name + "-up4_2", size=(4, 4), interpolation='bilinear')(up1)
        convo4_2 = Convo_Block(name=name + "-convo4_2", filters=nb_filters[5], **down_args)(convo4_2)

        convo1_2 = MaxPool2D((2, 2), 2, 'same')(convo1)
        convo1_2 = Convo_Block(name=name + "-convo1_2", filters=nb_filters[5], **down_args)(convo1_2)

        convo2_2 = Convo_Block(name=name + "-convo2_2", filters=nb_filters[5], **down_args)(convo2)

        up_convo3 = Conv2DTranspose(name=name + "-up_convo3", filters=nb_filters[5], **convo_trans_args)(up2)
        up3 = Concatenate(axis=3)([up_convo3, convo2_2, convo1_2, convo4_2, convo5_2])
        up3 = Convo_Block(name=name + "-up-3", filters=nb_filters[6], **down_args)(up3)

        convo5_1 = UpSampling2D(name=name + "-up5_1", size=(16, 16), interpolation='bilinear')(convo5)
        convo5_1 = Convo_Block(name=name + "-convo5_1", filters=nb_filters[5], **down_args)(convo5_1)

        convo4_1 = UpSampling2D(name=name + "-up4_1", size=(8, 8), interpolation='bilinear')(up1)
        convo4_1 = Convo_Block(name=name + "-convo4_1", filters=nb_filters[5], **down_args)(convo4_1)

        convo3_1 = UpSampling2D(name=name + "-up3_1", size=(4, 4), interpolation='bilinear')(up2)
        convo3_1 = Convo_Block(name=name + "-convo3_1", filters=nb_filters[5], **down_args)(convo3_1)

        convo1_1 = Convo_Block(name=name + "-convo1_1", filters=nb_filters[5], **down_args)(convo1)

        up_convo2 = Conv2DTranspose(name=name + "-up_convo2", filters=nb_filters[5], **convo_trans_args)(up3)
        up4 = Concatenate(axis=3)([up_convo2, convo1_1, convo5_1, convo4_1, convo3_1])
        up4 = Convo_Block(name=name + "-up-4", filters=nb_filters[6], **down_args)(up4)

        outconvo = Conv2D(name=name + "-final-convo", **out_args)(up4)
        sigmoid = Activation(tf.nn.sigmoid, dtype=tf.float32)
        if deep_supervision:
            side1 = UpSampling2D(name=name + "up-side1", size=(16, 16), interpolation='bilinear')(convo5)
            side1 = Conv2D(name=name + '-side1', **out_args)(side1)

            side2 = UpSampling2D(name=name + "up-side2", size=(8, 8), interpolation='bilinear')(up1)
            side2 = Conv2D(name=name + '-side2', **out_args)(side2)

            side3 = UpSampling2D(name=name + "up-side3", size=(4, 4), interpolation='bilinear')(up2)
            side3 = Conv2D(name=name + '-side3', **out_args)(side3)

            side4 = UpSampling2D(name=name + "up-side4", size=(2, 2), interpolation='bilinear')(up3)
            side4 = Conv2D(name=name + '-side4', **out_args)(side4)

            if cgm:
                cls = Dropout(rate=cgm_dropout)(convo5)
                cls = Conv2D(filters=2, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(cls)
                cls = GlobalMaxPool2D()(cls)
                cls = sigmoid(cls)
                cls = K.backend.max(cls, axis=-1)

                outconvo = multiply([outconvo, cls])
                side1 = multiply([side1, cls])
                side2 = multiply([side2, cls])
                side3 = multiply([side3, cls])
                side4 = multiply([side4, cls])
            return tf.stack([sigmoid(outconvo), sigmoid(side1), sigmoid(side2), sigmoid(side3), sigmoid(side4)])
        return sigmoid(outconvo)

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.kernel_regularizer = kernel_regularizer
    model.deep_supervision = deep_supervision
    model.classification_guided_module = cgm
    model.cgm_dropout = cgm_dropout
    return model


"""
Custom model that:
    * Extends the Unet3+ architecture on the decoder side
    * Adds non-standard skip connections on the decoder side
"""
def UNetExpTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
              name="UNetEXPTF",
              dropout=0.0,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=None,
              use_learnable_pool=False,
              deep_supervision=False,
              cgm=False,
              cgm_dropout=0.1,
              architecture="vgg",
              freeze="freeze",
              **kwargs):
    """
    Custom UNetExp Network:

    Args:
        input_shape (list): expected input shape by this network
        name: Name given to this network
        dropout (float): applied dropout amount at train time
        kernel_init (str): Kernel initialization mode
        normalize (bool): whether to apply Batch Normalization
        kernel_regularizer (fct): Kernel regularization function
        use_learnable_pool (bool): If True Let the network learn an interpolation between avg pool and max pool
        deep_supervision (bool): Whether to use intermediate outputs to compute the loss
        cgm (bool): whether to use the "classification_guided_module"
        cgm_dropout (bool): Amount of dropout if using the "classification_guided_module"
        architecture (str): Allows to specify backbone such as vgg architecture
    """
    #K.mixed_precision.set_global_policy('mixed_float16')
    def __build_model(inputs):
        nb_filters = [32, 64, 128, 256, 512, 64, 320]

        down_args = {
            'dropout': dropout,
            'kernel_init': kernel_init,
            'normalize': normalize,
            'kernel_regularizer': kernel_regularizer,
            'activation': tf.nn.silu
        }

        out_args = {
            'filters': 1,
            'kernel_size': (3, 3),
            'padding': 'same',
            'kernel_initializer': kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        convo_trans_args = {
            'kernel_size': (4, 4),
            'strides': (2, 2),
            'padding': 'same',
            'kernel_initializer': kernel_init,
            'kernel_regularizer': kernel_regularizer
        }
        nb_filters_concat = [32, 128, 256, 448]
        pretrained = None

        base_model = None
        if architecture == "vgg":
            inputs = K.applications.vgg19.preprocess_input(tf.cast(inputs,dtype=tf.float32))
            layer_names = ['block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
            base_model = K.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        elif architecture == "resnet50":
            inputs = K.applications.resnet_v2.preprocess_input(tf.cast(inputs,dtype=tf.float32))
            layer_names = ['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']
            base_model = K.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
        elif architecture == "resnet152":
            inputs = K.applications.resnet_v2.preprocess_input(tf.cast(inputs,dtype=tf.float32))
            layer_names = ['conv1_conv', 'conv2_block3_1_conv', 'conv3_block8_1_conv', 'conv4_block36_1_conv']
            base_model = K.applications.ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
        elif architecture == "resnet101":
            inputs = K.applications.resnet_v2.preprocess_input(tf.cast(inputs,dtype=tf.float32))
            layer_names = ['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block23_1_conv']
            base_model = K.applications.ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape)
        elif architecture is None:
            inputs = K.applications.vgg19.preprocess_input(tf.cast(inputs,dtype=tf.float32))

        if base_model is not None:
            # Full freeze of the base model - no data specific learned/only better gradient
            if freeze == "full-freeze":
                for layer in base_model.layers:
                    layer.trainable=False
            # BatchNorm can become trainable to learn data specifics
            elif freeze == "freeze":
                for layer in base_model.layers:
                    if isinstance(layer,K.layers.BatchNormalization):
                        layer.trainable = True
                    else:
                        layer.trainable = False
            # All layers can be unfrozen for fine-tunning ALREADY TRAINED models
            else:
                for layer in base_model.layers:
                    layer.trainable=True
            outputs = [base_model.get_layer(name).output for name in layer_names]
            model = K.Model([base_model.input], outputs)
            pretrained = model(inputs)
            # Fully add transfer layers
            nb_filters_concat = [96, 192, 320, 512]

        convo1, pool1 = Down_Block(name=name + "-down-block-1", filters=nb_filters[0], **down_args)(inputs)

        if architecture is not None:
            pretrained[0] = Convo_Block(name=name + "maxpool_vgg_1", filters=nb_filters[1], **down_args)(pretrained[0])
            pool1 = Concatenate(axis=3)([pool1, pretrained[0]])

        pool1 = Convo_Block(name=name + "pool1", filters=nb_filters_concat[0], **down_args)(pool1)

        convo2, pool2 = Down_Block(name=name + "-down-block-2", filters=nb_filters[1], **down_args)(pool1)

        pool0_1 = MaxPool2D((4, 4), 4, 'same')(convo1)
        pool0_1 = Convo_Block(name=name + "maxpool0_1", filters=nb_filters[1], **down_args)(pool0_1)

        pool1_1 = Convo_Block(name=name + "maxpool1_1", filters=nb_filters[1], **down_args)(pool2)

        pool2 = Concatenate(axis=3)([pool1_1, pool0_1])

        if architecture is not None:
            pretrained[1] = Convo_Block(name=name + "maxpool_vgg_2", filters=nb_filters[1], **down_args)(pretrained[1])
            pool2 = Concatenate(axis=3)([pool2, pretrained[1]])

        pool2 = Convo_Block(name=name + "pool2", filters=nb_filters_concat[1], **down_args)(pool2)

        convo3, pool3 = Down_Block(name=name + "-down-block-3", filters=nb_filters[2], **down_args)(pool2)

        pool0_2 = MaxPool2D((8, 8), 8, 'same')(convo1)
        pool0_2 = Convo_Block(name=name + "maxpool0_2", filters=nb_filters[1], **down_args)(pool0_2)
        pool1_2 = MaxPool2D((4, 4), 4, 'same')(convo2)
        pool1_2 = Convo_Block(name=name + "maxpool1_2", filters=nb_filters[1], **down_args)(pool1_2)

        pool2_2 = Convo_Block(name=name + "maxpool2_2", filters=nb_filters[2], **down_args)(pool3)

        pool3 = Concatenate(axis=3)([pool2_2, pool0_2, pool1_2])

        if architecture is not None:
            pretrained[2] = Convo_Block(name=name + "maxpool_vgg_3", filters=nb_filters[1], **down_args)(pretrained[2])
            pool3 = Concatenate(axis=3)([pool3, pretrained[2]])

        pool3 = Convo_Block(name=name + "pool3", filters=nb_filters_concat[2], **down_args)(pool3)

        convo4, pool4 = Down_Block(name=name + "-down-block-4", filters=nb_filters[3], **down_args)(pool3)

        pool0_3 = MaxPool2D((16, 16), 16, 'same')(convo1)
        pool0_3 = Convo_Block(name=name + "maxpool0_3", filters=nb_filters[1], **down_args)(pool0_3)
        pool1_3 = MaxPool2D((8, 8), 8, 'same')(convo2)
        pool1_3 = Convo_Block(name=name + "maxpool1_3", filters=nb_filters[1], **down_args)(pool1_3)
        pool2_3 = MaxPool2D((4, 4), 4, 'same')(convo3)
        pool2_3 = Convo_Block(name=name + "maxpool2_3", filters=nb_filters[1], **down_args)(pool2_3)

        pool3_3 = Convo_Block(name=name + "maxpool3_3", filters=nb_filters[3], **down_args)(pool4)

        pool4 = Concatenate(axis=3)([pool3_3, pool0_3, pool1_3, pool2_3])

        if architecture is not None:
            pretrained[3] = Convo_Block(name=name + "maxpool_vgg_4", filters=nb_filters[1], **down_args)(pretrained[3])
            pool4 = Concatenate(axis=3)([pool4, pretrained[3]])

        pool4 = Convo_Block(name=name + "pool4", filters=nb_filters_concat[3], **down_args)(pool4)

        convo5 = Convo_Block(name=name + "-convo-block", filters=nb_filters[4], **down_args)(pool4)

        convo3_4 = MaxPool2D((2, 2), 2, 'same')(convo3)
        convo3_4 = Convo_Block(name=name + "-convo3_4", filters=nb_filters[5], **down_args)(convo3_4)

        convo2_4 = MaxPool2D((4, 4), 4, 'same')(convo2)
        convo2_4 = Convo_Block(name=name + "-convo2_4", filters=nb_filters[5], **down_args)(convo2_4)

        convo1_4 = MaxPool2D((8, 8), 8, 'same')(convo1)
        convo1_4 = Convo_Block(name=name + "-convo1_4", filters=nb_filters[5], **down_args)(convo1_4)

        convo4_4 = Convo_Block(name=name + "-convo4_4", filters=nb_filters[5], **down_args)(convo4)

        up_convo5 = Conv2DTranspose(name=name + "-up_convo5", filters=nb_filters[5], **convo_trans_args)(convo5)
        up1 = Concatenate(axis=3)([up_convo5, convo4_4, convo3_4, convo2_4, convo1_4])
        up1 = Convo_Block(name=name + "-up-1", filters=nb_filters[6], **down_args)(up1)

        convo5_3 = UpSampling2D(name=name + "-up5_3", size=(4, 4), interpolation='bilinear')(convo5)
        convo5_3 = Convo_Block(name=name + "-convo5_3", filters=nb_filters[5], **down_args)(convo5_3)

        convo2_3 = MaxPool2D((2, 2), 2, 'same')(convo2)
        convo2_3 = Convo_Block(name=name + "-convo2_3", filters=nb_filters[5], **down_args)(convo2_3)

        convo1_3 = MaxPool2D((4, 4), 4, 'same')(convo1)
        convo1_3 = Convo_Block(name=name + "-convo1_3", filters=nb_filters[5], **down_args)(convo1_3)

        convo3_3 = Convo_Block(name=name + "-convo3_3", filters=nb_filters[5], **down_args)(convo3)

        up_convo4 = Conv2DTranspose(name=name + "-up_convo4", filters=nb_filters[5], **convo_trans_args)(up1)
        up2 = Concatenate(axis=3)([up_convo4, convo3_3, convo2_3, convo1_3, convo5_3])
        up2 = Convo_Block(name=name + "-up-2", filters=nb_filters[6], **down_args)(up2)

        convo5_2 = UpSampling2D(name=name + "-up5_2", size=(8, 8), interpolation='bilinear')(convo5)
        convo5_2 = Convo_Block(name=name + "-convo5_2", filters=nb_filters[5], **down_args)(convo5_2)

        convo4_2 = UpSampling2D(name=name + "-up4_2", size=(4, 4), interpolation='bilinear')(up1)
        convo4_2 = Convo_Block(name=name + "-convo4_2", filters=nb_filters[5], **down_args)(convo4_2)

        convo1_2 = MaxPool2D((2, 2), 2, 'same')(convo1)
        convo1_2 = Convo_Block(name=name + "-convo1_2", filters=nb_filters[5], **down_args)(convo1_2)

        convo2_2 = Convo_Block(name=name + "-convo2_2", filters=nb_filters[5], **down_args)(convo2)

        up_convo3 = Conv2DTranspose(name=name + "-up_convo3", filters=nb_filters[5], **convo_trans_args)(up2)
        up3 = Concatenate(axis=3)([up_convo3, convo2_2, convo1_2, convo4_2, convo5_2])
        up3 = Convo_Block(name=name + "-up-3", filters=nb_filters[6], **down_args)(up3)

        convo5_1 = UpSampling2D(name=name + "-up5_1", size=(16, 16), interpolation='bilinear')(convo5)
        convo5_1 = Convo_Block(name=name + "-convo5_1", filters=nb_filters[5], **down_args)(convo5_1)

        convo4_1 = UpSampling2D(name=name + "-up4_1", size=(8, 8), interpolation='bilinear')(up1)
        convo4_1 = Convo_Block(name=name + "-convo4_1", filters=nb_filters[5], **down_args)(convo4_1)

        convo3_1 = UpSampling2D(name=name + "-up3_1", size=(4, 4), interpolation='bilinear')(up2)
        convo3_1 = Convo_Block(name=name + "-convo3_1", filters=nb_filters[5], **down_args)(convo3_1)

        convo1_1 = Convo_Block(name=name + "-convo1_1", filters=nb_filters[5], **down_args)(convo1)

        up_convo2 = Conv2DTranspose(name=name + "-up_convo2", filters=nb_filters[5], **convo_trans_args)(up3)
        up4 = Concatenate(axis=3)([up_convo2, convo1_1, convo5_1, convo4_1, convo3_1])
        up4 = Convo_Block(name=name + "-up-4", filters=nb_filters[6], **down_args)(up4)

        outconvo = Conv2D(name=name + "-final-convo", **out_args)(up4)
        sigmoid = Activation(tf.nn.sigmoid, dtype=tf.float32)
        if deep_supervision:
            side1 = UpSampling2D(name=name + "up-side1", size=(16, 16), interpolation='bilinear')(convo5)
            side1 = Conv2D(name=name + '-side1', **out_args)(side1)

            side2 = UpSampling2D(name=name + "up-side2", size=(8, 8), interpolation='bilinear')(up1)
            side2 = Conv2D(name=name + '-side2', **out_args)(side2)

            side3 = UpSampling2D(name=name + "up-side3", size=(4, 4), interpolation='bilinear')(up2)
            side3 = Conv2D(name=name + '-side3', **out_args)(side3)

            side4 = UpSampling2D(name=name + "up-side4", size=(2, 2), interpolation='bilinear')(up3)
            side4 = Conv2D(name=name + '-side4', **out_args)(side4)

            if cgm:
                cls = Dropout(rate=cgm_dropout)(convo5)
                cls = Conv2D(filters=2, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(cls)
                cls = GlobalMaxPool2D()(cls)
                cls = sigmoid(cls)
                cls = K.backend.max(cls, axis=-1)

                outconvo = multiply([outconvo, cls])
                side1 = multiply([side1, cls])
                side2 = multiply([side2, cls])
                side3 = multiply([side3, cls])
                side4 = multiply([side4, cls])

            return tf.stack([sigmoid(outconvo), sigmoid(side1), sigmoid(side2), sigmoid(side3), sigmoid(side4)])
        return sigmoid(outconvo)

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.kernel_regularizer = kernel_regularizer
    model.deep_supervision = deep_supervision
    model.classification_guided_module = cgm
    model.cgm_dropout = cgm_dropout
    return model