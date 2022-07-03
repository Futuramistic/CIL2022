from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K
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

    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        down_args = {
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
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
            'kernel_size':(1,1),
            'padding':'same',
            'activation':'sigmoid',
            'kernel_initializer':kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        pool_fct = Down_Block_LearnablePool if use_learnable_pool else Down_Block

        convo1,pool1 = pool_fct(name=name+"-down-block-1",filters=nb_filters[0],kernel_size=5,**down_args)(inputs)
        convo2,pool2 = pool_fct(name=name+"-down-block-2",filters=nb_filters[1],kernel_size=5,**down_args)(pool1)
        convo3,pool3 = pool_fct(name=name+"-down-block-3",filters=nb_filters[2],**down_args)(pool2)
        convo4,pool4 = pool_fct(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

        convo5 = Convo_Block(name=name+"-convo-block",filters=nb_filters[4],dilation_rate=2,**down_args)(pool4)

        up1 = Up_Block(name=name+"-up-block-1",filters=nb_filters[3],**up_args)(x=convo5,merger=[convo4])
        up2 = Up_Block(name=name+"-up-block-2",filters=nb_filters[2],**up_args)(x=up1,   merger=[convo3])
        up3 = Up_Block(name=name+"-up-block-3",filters=nb_filters[1],**up_args)(x=up2,   merger=[convo2])
        up4 = Up_Block(name=name+"-up-block-4",filters=nb_filters[0],**up_args)(x=up3,   merger=[convo1])

        return Conv2D(name=name+"-final-convo",**out_args)(up4)
    
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
           up_transpose=True,
           kernel_regularizer=K.regularizers.l2(),
           use_learnable_pool=False,
           **kwargs):

    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        down_args = {
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
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
            'kernel_size':(1,1),
            'padding':'same',
            'activation':'sigmoid',
            'kernel_initializer':kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        convo_trans_args = {
            'kernel_size':(2, 2),
            'strides':(2, 2),
            'padding':'same',
            'kernel_initializer':kernel_init,
            'kernel_regularizer':kernel_regularizer
        }

        pool_fct = Down_Block_LearnablePool if use_learnable_pool else Down_Block

        convo1,pool1 = pool_fct(name=name+"-down-block-1",filters=nb_filters[0],**down_args)(inputs)
        convo2,pool2 = pool_fct(name=name+"-down-block-2",filters=nb_filters[1],**down_args)(pool1)
        convo3,pool3 = pool_fct(name=name+"-down-block-3",filters=nb_filters[2],**down_args)(pool2)
        convo4,pool4 = pool_fct(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

        convo5 = Convo_Block(name=name+"-convo-block",filters=nb_filters[4],**down_args)(pool4)

        convo3_4 = MaxPool2D((2,2),2,'same')(convo3)
        convo3_4 = Convo_Block(name=name+"-convo3_4",filters=nb_filters[3],**down_args)(convo3_4)

        convo2_4 = MaxPool2D((4,4),4,'same')(convo2)
        convo2_4 = Convo_Block(name=name+"-convo2_4",filters=nb_filters[3],**down_args)(convo2_4)

        convo1_4 = MaxPool2D((8,8),8,'same')(convo1)
        convo1_4 = Convo_Block(name=name+"-convo1_4",filters=nb_filters[3],**down_args)(convo1_4)

        up_convo5 = Conv2DTranspose(name=name+"-up_convo5",filters=nb_filters[3],**convo_trans_args)(convo5)
        up1 = Concatenate(axis=3)([up_convo5,convo4,convo3_4,convo2_4,convo1_4])
        up1 = Convo_Block(name=name+"-up-1",filters=nb_filters[3],**down_args)(up1)

        convo5_3 = UpSampling2D(name=name+"-up5_3",size=(4,4),interpolation='bilinear')(convo5)
        convo5_3 = Convo_Block(name=name+"-convo5_3",filters=nb_filters[2],**down_args)(convo5_3)

        convo2_3 = MaxPool2D((2,2),2,'same')(convo2)
        convo2_3 = Convo_Block(name=name+"-convo2_3",filters=nb_filters[2],**down_args)(convo2_3)

        convo1_3 = MaxPool2D((4,4),4,'same')(convo1)
        convo1_3 = Convo_Block(name=name+"-convo1_3",filters=nb_filters[2],**down_args)(convo1_3)


        up_convo4 = Conv2DTranspose(name=name+"-up_convo4",filters=nb_filters[2],**convo_trans_args)(up1)
        up2 = Concatenate(axis=3)([up_convo4,convo3,convo2_3,convo1_3,convo5_3])
        up2 = Convo_Block(name=name+"-up-2",filters=nb_filters[2],**down_args)(up2)

        convo5_2 = UpSampling2D(name=name+"-up5_2",size=(8,8),interpolation='bilinear')(convo5)
        convo5_2 = Convo_Block(name=name+"-convo5_2",filters=nb_filters[1],**down_args)(convo5_2)

        convo4_2 = UpSampling2D(name=name+"-up4_2",size=(4,4),interpolation='bilinear')(up1)
        convo4_2 = Convo_Block(name=name+"-convo4_2",filters=nb_filters[1],**down_args)(convo4_2)

        convo1_2 = MaxPool2D((2,2),2,'same')(convo1)
        convo1_2 = Convo_Block(name=name+"-convo1_2",filters=nb_filters[1],**down_args)(convo1_2)

        up3 = Up_Block(name=name+"-up-block-3",filters=nb_filters[1],**up_args)(x=up2,   merger=[convo2,convo1_2,convo4_2,convo5_2])

        convo5_1 = UpSampling2D(name=name+"-up5_1",size=(16,16),interpolation='bilinear')(convo5)
        convo5_1 = Convo_Block(name=name+"-convo5_1",filters=nb_filters[0],**down_args)(convo5_1)

        convo4_1 = UpSampling2D(name=name+"-up4_1",size=(8,8),interpolation='bilinear')(up1)
        convo4_1 = Convo_Block(name=name+"-convo4_1",filters=nb_filters[0],**down_args)(convo4_1)

        convo3_1 = UpSampling2D(name=name+"-up3_1",size=(4,4),interpolation='bilinear')(up2)
        convo3_1 = Convo_Block(name=name+"-convo3_1",filters=nb_filters[0],**down_args)(convo3_1)

        up4 = Up_Block(name=name+"-up-block-4",filters=nb_filters[0],**up_args)(x=up3,  merger=[convo1,convo5_1,convo4_1,convo3_1])

        return Conv2D(name=name+"-final-convo",**out_args)(up4)
    
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


# Extend the Unet3+ architecture on the decoder side
# * Added non-standard skip connections on the decoder side
def UNetExpTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
           name="UNetEXPTF",
           dropout=0.5,
           kernel_init='he_normal',
           normalize=True,
           up_transpose=True,
           kernel_regularizer=K.regularizers.l2(),
           use_learnable_pool=False,
           **kwargs):

    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        down_args = {
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
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
            'kernel_size':(1,1),
            'padding':'same',
            'activation':'sigmoid',
            'kernel_initializer':kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        convo_trans_args = {
            'kernel_size':(2, 2),
            'strides':(2, 2),
            'padding':'same',
            'kernel_initializer':kernel_init,
            'kernel_regularizer':kernel_regularizer
        }

        pool_fct = Down_Block_LearnablePool if use_learnable_pool else Down_Block

        convo1,pool1 = pool_fct(name=name+"-down-block-1",filters=nb_filters[0],**down_args)(inputs)
        convo2,pool2 = pool_fct(name=name+"-down-block-2",filters=nb_filters[1],**down_args)(pool1)

        pool0_1 = MaxPool2D((4,4),4,'same')(convo1)
        pool0_1 = Convo_Block(name=name+"maxpool0_1", filters=nb_filters[1], **down_args)(pool0_1)

        pool2 = Concatenate(axis=3)([pool2,pool0_1])
        pool2 = Convo_Block(name=name+"pool2", filters=nb_filters[1], **down_args)(pool2)

        convo3,pool3 = pool_fct(name=name+"-down-block-3",filters=nb_filters[2],**down_args)(pool2)

        pool0_2 = MaxPool2D((8,8),8,'same')(convo1)
        pool0_2 = Convo_Block(name=name+"maxpool0_2", filters=nb_filters[2], **down_args)(pool0_2)
        pool1_2 = MaxPool2D((4,4),4,'same')(convo2)
        pool1_2 = Convo_Block(name=name+"maxpool1_2", filters=nb_filters[2], **down_args)(pool1_2)

        pool3 = Concatenate(axis=3)([pool3,pool0_2,pool1_2])
        pool3 = Convo_Block(name=name+"pool3", filters=nb_filters[2], **down_args)(pool3)

        convo4,pool4 = pool_fct(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

        pool0_3 = MaxPool2D((16,16),16,'same')(convo1)
        pool0_3 = Convo_Block(name=name+"maxpool0_3", filters=nb_filters[3], **down_args)(pool0_3)
        pool1_3 = MaxPool2D((8,8),8,'same')(convo2)
        pool1_3 = Convo_Block(name=name+"maxpool1_3", filters=nb_filters[3], **down_args)(pool1_3)
        pool2_3 = MaxPool2D((4,4),4,'same')(convo3)
        pool2_3 = Convo_Block(name=name+"maxpool2_3", filters=nb_filters[3], **down_args)(pool2_3)

        pool4 = Concatenate(axis=3)([pool4,pool0_3,pool1_3,pool2_3])
        pool4 = Convo_Block(name=name+"pool4", filters=nb_filters[3], **down_args)(pool4)

        convo5 = Convo_Block(name=name+"-convo-block",filters=nb_filters[4],**down_args)(pool4)

        convo3_4 = MaxPool2D((2,2),2,'same')(convo3)
        convo3_4 = Convo_Block(name=name+"-convo3_4",filters=nb_filters[3],**down_args)(convo3_4)

        convo2_4 = MaxPool2D((4,4),4,'same')(convo2)
        convo2_4 = Convo_Block(name=name+"-convo2_4",filters=nb_filters[3],**down_args)(convo2_4)

        convo1_4 = MaxPool2D((8,8),8,'same')(convo1)
        convo1_4 = Convo_Block(name=name+"-convo1_4",filters=nb_filters[3],**down_args)(convo1_4)

        up_convo5 = Conv2DTranspose(name=name+"-up_convo5",filters=nb_filters[3],**convo_trans_args)(convo5)
        up1 = Concatenate(axis=3)([up_convo5,convo4,convo3_4,convo2_4,convo1_4])
        up1 = Convo_Block(name=name+"-up-1",filters=nb_filters[3],**down_args)(up1)

        convo5_3 = UpSampling2D(name=name+"-up5_3",size=(4,4),interpolation='bilinear')(convo5)
        convo5_3 = Convo_Block(name=name+"-convo5_3",filters=nb_filters[2],**down_args)(convo5_3)

        convo2_3 = MaxPool2D((2,2),2,'same')(convo2)
        convo2_3 = Convo_Block(name=name+"-convo2_3",filters=nb_filters[2],**down_args)(convo2_3)

        convo1_3 = MaxPool2D((4,4),4,'same')(convo1)
        convo1_3 = Convo_Block(name=name+"-convo1_3",filters=nb_filters[2],**down_args)(convo1_3)


        up_convo4 = Conv2DTranspose(name=name+"-up_convo4",filters=nb_filters[2],**convo_trans_args)(up1)
        up2 = Concatenate(axis=3)([up_convo4,convo3,convo2_3,convo1_3,convo5_3])
        up2 = Convo_Block(name=name+"-up-2",filters=nb_filters[2],**down_args)(up2)

        convo5_2 = UpSampling2D(name=name+"-up5_2",size=(8,8),interpolation='bilinear')(convo5)
        convo5_2 = Convo_Block(name=name+"-convo5_2",filters=nb_filters[1],**down_args)(convo5_2)

        convo4_2 = UpSampling2D(name=name+"-up4_2",size=(4,4),interpolation='bilinear')(up1)
        convo4_2 = Convo_Block(name=name+"-convo4_2",filters=nb_filters[1],**down_args)(convo4_2)

        convo1_2 = MaxPool2D((2,2),2,'same')(convo1)
        convo1_2 = Convo_Block(name=name+"-convo1_2",filters=nb_filters[1],**down_args)(convo1_2)

        up3 = Up_Block(name=name+"-up-block-3",filters=nb_filters[1],**up_args)(x=up2,   merger=[convo2,convo1_2,convo4_2,convo5_2])

        convo5_1 = UpSampling2D(name=name+"-up5_1",size=(16,16),interpolation='bilinear')(convo5)
        convo5_1 = Convo_Block(name=name+"-convo5_1",filters=nb_filters[0],**down_args)(convo5_1)

        convo4_1 = UpSampling2D(name=name+"-up4_1",size=(8,8),interpolation='bilinear')(up1)
        convo4_1 = Convo_Block(name=name+"-convo4_1",filters=nb_filters[0],**down_args)(convo4_1)

        convo3_1 = UpSampling2D(name=name+"-up3_1",size=(4,4),interpolation='bilinear')(up2)
        convo3_1 = Convo_Block(name=name+"-convo3_1",filters=nb_filters[0],**down_args)(convo3_1)

        up4 = Up_Block(name=name+"-up-block-4",filters=nb_filters[0],**up_args)(x=up3,  merger=[convo1,convo5_1,convo4_1,convo3_1])

        return Conv2D(name=name+"-final-convo",**out_args)(up4)
    
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