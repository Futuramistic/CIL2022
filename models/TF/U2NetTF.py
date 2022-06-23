from .blocks import *
from utils import *
import keras

## Layers of U^2Net - quite big; easier to add them here
class RSU7(keras.layers.Layer):

    def __init__(self,
              dropout=0.5,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              mid_channels = 12,
              out_channels = 3,
              **kwargs):
            super(RSU7, self).__init__()

            mid_args = {
                'filters':mid_channels,
                'dropout': dropout,
                'kernel_init':kernel_init,
                'normalize':normalize,
                'kernel_regularizer': kernel_regularizer
            }

            out_args = {
                'dropout':dropout,
                'filters':out_channels,
                'kernel_init':kernel_init,
                'normalize':normalize,
                'kernel_regularizer':kernel_regularizer,
            }

            pool_args = {
                'pool_size':(2,2),
                'strides':2,
                'padding':'same'
            }

            up_args = {
                'size':(2,2),
                'interpolation':'bilinear'
            }

            self.convo_in = ConvoRelu_Block(**out_args)

            self.x1 =ConvoRelu_Block(**mid_args)
            self.pool1 = MaxPool2D(**pool_args)

            self.x2 = ConvoRelu_Block(**mid_args)
            self.pool2 = MaxPool2D(**pool_args)

            self.x3 = ConvoRelu_Block(**mid_args)
            self.pool3 = MaxPool2D(**pool_args)

            self.x4 = ConvoRelu_Block(**mid_args)
            self.pool4 = MaxPool2D(**pool_args)

            self.x5 = ConvoRelu_Block(**mid_args)
            self.pool5 = MaxPool2D(**pool_args)

            self.x6 = ConvoRelu_Block(**mid_args)
            self.x7 = ConvoRelu_Block(dilation_rate=2,**mid_args)

            self.x6d = ConvoRelu_Block(**mid_args)
            self.x6up = UpSampling2D(**up_args)

            self.x5d = ConvoRelu_Block(**mid_args)
            self.x5up = UpSampling2D(**up_args)

            self.x4d = ConvoRelu_Block(**mid_args)
            self.x4up = UpSampling2D(**up_args)

            self.x3d = ConvoRelu_Block(**mid_args)
            self.x3up = UpSampling2D(**up_args)
            
            self.x2d = ConvoRelu_Block(**mid_args)
            self.x2up = UpSampling2D(**up_args)

            self.x1d = ConvoRelu_Block(**out_args)

    def call(self, inputs,**kwargs):
        x = inputs
        hx_in = self.convo_in(x,**kwargs)

        hx1 = self.x1(hx_in,**kwargs)
        hx = self.pool1(hx1)

        hx2 = self.x2(hx,**kwargs)
        hx = self.pool2(hx2)

        hx3 = self.x3(hx,**kwargs)
        hx = self.pool3(hx3)

        hx4 = self.x4(hx,**kwargs)
        hx = self.pool4(hx4)

        hx5 = self.x5(hx,**kwargs)
        hx = self.pool5(hx5)

        hx6 = self.x6(hx,**kwargs)

        hx7 = self.x7(hx6,**kwargs)

        hx6d = self.x6d(tf.concat([hx7, hx6], axis=3),**kwargs)
        hx6dup = self.x6up(hx6d)

        hx5d = self.x5d(tf.concat([hx6dup, hx5], axis=3),**kwargs)
        hx5dup = self.x5up(hx5d)

        hx4d = self.x4d(tf.concat([hx5dup, hx4], axis=3),**kwargs)
        hx4dup = self.x4up(hx4d)

        hx3d = self.x3d(tf.concat([hx4dup, hx3], axis=3),**kwargs)
        hx3dup = self.x3up(hx3d)

        hx2d =  self.x2d(tf.concat([hx3dup, hx2], axis=3),**kwargs)
        hx2dup = self.x2up(hx2d)

        hx1d = self.x1d(tf.concat([hx2dup, hx1], axis=3),**kwargs)
        
        return hx1d + hx_in

class RSU6(keras.layers.Layer):
    def __init__(self,
              dropout=0.,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              mid_channels = 12,
              out_channels = 3,
              **kwargs):

        super(RSU6, self).__init__()

        mid_args = {
            'filters':mid_channels,
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'dropout':dropout,
            'filters':out_channels,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer':kernel_regularizer,
        }

        pool_args = {
            'pool_size':(2,2),
            'strides':2,
            'padding':'same'
        }

        up_args = {
            'size':(2,2),
            'interpolation':'bilinear'
        }

        self.convo_in = ConvoRelu_Block(**out_args)

        self.x1 = ConvoRelu_Block(**mid_args)
        self.pool1   = MaxPool2D(**pool_args)

        self.x2 = ConvoRelu_Block(**mid_args)
        self.pool2   = MaxPool2D(**pool_args)

        self.x3 = ConvoRelu_Block(**mid_args)
        self.pool3   = MaxPool2D(**pool_args)

        self.x4 = ConvoRelu_Block(**mid_args)
        self.pool4   = MaxPool2D(**pool_args)

        self.x5 = ConvoRelu_Block(**mid_args)
        self.pool5   = MaxPool2D(**pool_args)

        self.x6 = ConvoRelu_Block(dilation_rate=2,**mid_args)

        self.x5d = ConvoRelu_Block(**mid_args)
        self.x1up = UpSampling2D(**up_args)
        self.x4d = ConvoRelu_Block(**mid_args)
        self.x2up = UpSampling2D(**up_args)
        self.x3d = ConvoRelu_Block(**mid_args)
        self.x3up = UpSampling2D(**up_args)
        self.x2d = ConvoRelu_Block(**mid_args)
        self.x4up = UpSampling2D(**up_args)
        self.x1d = ConvoRelu_Block(**out_args)
    
    def call(self, inputs,**kwargs):
        hx = inputs
        hxin = self.convo_in(hx,**kwargs)

        hx1 = self.x1(hxin,**kwargs)
        hx = self.pool1(hx1)

        hx2 = self.x2(hx,**kwargs)
        hx = self.pool2(hx2)

        hx3 = self.x3(hx,**kwargs)
        hx = self.pool3(hx3)

        hx4 = self.x4(hx,**kwargs)
        hx = self.pool4(hx4)

        hx5 = self.x5(hx,**kwargs)

        hx6 = self.x6(hx5,**kwargs)

        hx5d = self.x5d(tf.concat([hx6, hx5], axis=3),**kwargs)
        hx5dup = self.x4up(hx5d)

        hx4d = self.x4d(tf.concat([hx5dup, hx4], axis=3),**kwargs)
        hx4dup = self.x3up(hx4d)

        hx3d = self.x3d(tf.concat([hx4dup, hx3], axis=3),**kwargs)
        hx3dup = self.x2up(hx3d)

        hx2d =  self.x2d(tf.concat([hx3dup, hx2], axis=3),**kwargs)
        hx2dup = self.x1up(hx2d)

        hx1d = self.x1d(tf.concat([hx2dup, hx1], axis=3),**kwargs)
        
        return hx1d + hxin

class RSU5(keras.layers.Layer):
    def __init__(self,
              dropout=0.,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              mid_channels = 12,
              out_channels = 3,
              **kwargs):

        super(RSU5, self).__init__()

        mid_args = {
            'filters':mid_channels,
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'dropout':dropout,
            'filters':out_channels,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer':kernel_regularizer,
        }

        pool_args = {
            'pool_size':(2,2),
            'strides':2,
            'padding':'same'
        }

        up_args = {
            'size':(2,2),
            'interpolation':'bilinear'
        }

        self.x0 = ConvoRelu_Block(**out_args)

        self.x1 = ConvoRelu_Block(**mid_args)
        self.pool1   = MaxPool2D(**pool_args)

        self.x2 = ConvoRelu_Block(**mid_args)
        self.pool2   = MaxPool2D(**pool_args)

        self.x3 = ConvoRelu_Block(**mid_args)
        self.pool3   = MaxPool2D(**pool_args)

        self.x4 = ConvoRelu_Block(**mid_args)
        self.pool4   = MaxPool2D(**pool_args)

        self.x5 = ConvoRelu_Block(dilation_rate=2,**mid_args)

        self.x4d = ConvoRelu_Block(**mid_args)
        self.x1up = UpSampling2D(**up_args)
        self.x3d = ConvoRelu_Block(**mid_args)
        self.x2up = UpSampling2D(**up_args)
        self.x2d = ConvoRelu_Block(**mid_args)
        self.x3up = UpSampling2D(**up_args)
        self.x1d = ConvoRelu_Block(**out_args)
    
    def call(self, inputs,**kwargs):
        hx = inputs
        hxin = self.x0(hx,**kwargs)

        hx1 = self.x1(hxin,**kwargs)
        hx = self.pool1(hx1)

        hx2 = self.x2(hx,**kwargs)
        hx = self.pool2(hx2)

        hx3 = self.x3(hx,**kwargs)
        hx = self.pool3(hx3)

        hx4 = self.x4(hx,**kwargs)

        hx5 = self.x5(hx4,**kwargs)

        hx4d = self.x4d(tf.concat([hx5, hx4], axis=3),**kwargs)
        hx4dup = self.x3up(hx4d)

        hx3d = self.x3d(tf.concat([hx4dup, hx3], axis=3),**kwargs)
        hx3dup = self.x2up(hx3d)

        hx2d =  self.x2d(tf.concat([hx3dup, hx2], axis=3),**kwargs)
        hx2dup = self.x1up(hx2d)

        hx1d = self.x1d(tf.concat([hx2dup, hx1], axis=3),**kwargs)
        
        return hx1d + hxin


class RSU4(keras.layers.Layer):
    def __init__(self,
              dropout=0.,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              mid_channels = 12,
              out_channels = 3,
              **kwargs):

        super(RSU4, self).__init__()

        mid_args = {
            'filters':mid_channels,
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'dropout':dropout,
            'filters':out_channels,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer':kernel_regularizer,
        }

        pool_args = {
            'pool_size':(2,2),
            'strides':2,
            'padding':'same'
        }

        up_args = {
            'size':(2,2),
            'interpolation':'bilinear'
        }

        self.x0 = ConvoRelu_Block(**out_args)

        self.x1 = ConvoRelu_Block(**mid_args)
        self.pool1   = MaxPool2D(**pool_args)

        self.x2 = ConvoRelu_Block(**mid_args)
        self.pool2   = MaxPool2D(**pool_args)

        self.x3 = ConvoRelu_Block(**mid_args)
        self.pool3   = MaxPool2D(**pool_args)

        self.x4 = ConvoRelu_Block(dilation_rate=2,**mid_args)

        self.x3d = ConvoRelu_Block(**mid_args)
        self.x1up = UpSampling2D(**up_args)
        self.x2d = ConvoRelu_Block(**mid_args)
        self.x2up = UpSampling2D(**up_args)
        self.x1d = ConvoRelu_Block(**out_args)
    
    def call(self, inputs,**kwargs):
        hx = inputs
        hxin = self.x0(hx,**kwargs)

        hx1 = self.x1(hxin,**kwargs)
        hx = self.pool1(hx1)

        hx2 = self.x2(hx,**kwargs)
        hx = self.pool2(hx2)

        hx3 = self.x3(hx,**kwargs)

        hx4 = self.x4(hx3,**kwargs)

        hx3d = self.x3d(tf.concat([hx4, hx3], axis=3),**kwargs)
        hx3dup = self.x2up(hx3d)

        hx2d =  self.x2d(tf.concat([hx3dup, hx2], axis=3),**kwargs)
        hx2dup = self.x1up(hx2d)

        hx1d = self.x1d(tf.concat([hx2dup, hx1], axis=3),**kwargs)
        
        return hx1d + hxin

class RSU4F(keras.layers.Layer):
    def __init__(self,
              dropout=0.,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              mid_channels = 12,
              out_channels = 3,
              **kwargs):
        super(RSU4F, self).__init__()

        mid_args = {
            'filters':mid_channels,
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'dropout':dropout,
            'filters':out_channels,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer':kernel_regularizer,
        }

        self.x0 =   ConvoRelu_Block(dilation_rate=1,**out_args)
        self.x1 =   ConvoRelu_Block(dilation_rate=1,**mid_args)
        self.x2 =   ConvoRelu_Block(dilation_rate=2,**mid_args)
        self.x3 =   ConvoRelu_Block(dilation_rate=4,**mid_args)
        self.x4 =   ConvoRelu_Block(dilation_rate=8,**mid_args)
        self.x3d =  ConvoRelu_Block(dilation_rate=4,**mid_args)
        self.x2d =  ConvoRelu_Block(dilation_rate=2,**mid_args)
        self.x1d =  ConvoRelu_Block(dilation_rate=1,**out_args)
    
    def call(self, inputs,**kwargs):
        hx = inputs
        hxin = self.x0(hx,**kwargs)
        
        hx1 = self.x1(hxin,**kwargs)
        hx2 = self.x2(hx1,**kwargs)
        hx3 = self.x3(hx2,**kwargs)
        hx4 = self.x4(hx3,**kwargs)
        hx3d = self.x3d(tf.concat([hx4, hx3], axis=3),**kwargs)
        hx2d = self.x2d(tf.concat([hx3d, hx2], axis=3),**kwargs)
        hx1d = self.x1d(tf.concat([hx2d, hx1], axis=3),**kwargs)
        return hx1d + hxin

## Original U^2Net
def U2NetTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
              name="U2NetTF",
              dropout=0.5,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              **kwargs):

    network_args = {
        'dropout': dropout,
        'kernel_init':kernel_init,
        'normalize':normalize,
        'kernel_regularizer': kernel_regularizer
    }

    pool_args = {
        'pool_size':(2,2),
        'strides':2,
        'padding':'same'
    }

    convo2d_args={
        'filters':1, 
        'kernel_size':(3, 3),
        'kernel_regularizer': kernel_regularizer,
        'padding':'same'
    }

    up_args = {
        'size':(2,2),
        'interpolation':'bilinear'
    }

    def __build_model(inputs):

        hx1 = RSU7(mid_channels=32,out_channels=64,**network_args)(inputs)
        hx = MaxPool2D(**pool_args)(hx1)

        hx2 = RSU6(mid_channels=32,out_channels=128,**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx2)

        hx3 = RSU5(mid_channels=64,out_channels=256,**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx3)

        hx4 = RSU4(mid_channels=128,out_channels=512,**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx4)

        hx5 = RSU4F(mid_channels=256,out_channels=512,**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx5)

        hx6 = RSU4F(mid_channels=256,out_channels=512,**network_args)(hx)
        hx6dup = UpSampling2D(**up_args)(hx6)
        side6_in = Conv2D(**convo2d_args)(hx6)
        side6 = UpSampling2D(size=(32, 32), interpolation='bilinear')(side6_in)

        hx5d = RSU4F(mid_channels=256,out_channels=512,**network_args)(tf.concat([hx6dup, hx5], axis=3))
        hx5dup = UpSampling2D(**up_args)(hx5d)
        side5_in = Conv2D(**convo2d_args)(hx5d)
        side5 = UpSampling2D(size=(16, 16), interpolation='bilinear')(side5_in)

        hx4d = RSU4(mid_channels=128,out_channels=256,**network_args)(tf.concat([hx5dup, hx4], axis=3))
        hx4dup = UpSampling2D(**up_args)(hx4d)
        side4_in = Conv2D(**convo2d_args)(hx4d)
        side4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(side4_in)

        hx3d = RSU5(mid_channels=64,out_channels=128,**network_args)(tf.concat([hx4dup, hx3], axis=3))
        hx3dup = UpSampling2D(**up_args)(hx3d)
        side3_in = Conv2D(**convo2d_args)(hx3d)
        side3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(side3_in)

        hx2d = RSU6(mid_channels=32,out_channels=64,**network_args)(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = UpSampling2D(**up_args)(hx2d)
        side2_in = Conv2D(**convo2d_args)(hx2d)
        side2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(side2_in)
        
        hx1d = RSU7(mid_channels=16,out_channels=64,**network_args)(tf.concat([hx2dup, hx1], axis=3))
        side1 = Conv2D(**convo2d_args)(hx1d)
        outconv = Conv2D(1, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(tf.concat([side1, side2, side3, side4, side5, side6], axis=3))

        sigmoid = keras.activations.sigmoid
        return tf.stack([sigmoid(outconv), sigmoid(side1), sigmoid(side2), sigmoid(side3), sigmoid(side4), sigmoid(side5), sigmoid(side6)])
        #return sigmoid(outconv)
    
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name='U2NetTF')
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.up_transpose = None
    model.kernel_regularizer = kernel_regularizer
    return model

## U^2Net (small)
def U2NetSmallTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
              name="U2NetTF",
              dropout=0.5,
              kernel_init='he_normal',
              normalize=True,
              kernel_regularizer=K.regularizers.l2(),
              **kwargs):

    network_args = {
        'dropout': dropout,
        'kernel_init':kernel_init,
        'normalize':normalize,
        'kernel_regularizer': kernel_regularizer,
        'mid_channels':16,
        'out_channels':64
    }

    pool_args = {
        'pool_size':(2,2),
        'strides':2,
        'padding':'same'
    }

    convo2d_args={
        'filters':1, 
        'kernel_size':(3, 3), 
        'padding':'same'
    }

    up_args = {
        'size':(2,2),
        'interpolation':'bilinear'
    }

    def __build_model(inputs):

        hx1 = RSU7(**network_args)(inputs)
        hx = MaxPool2D(**pool_args)(hx1)

        hx2 = RSU6(**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx2)

        hx3 = RSU5(**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx3)

        hx4 = RSU4(**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx4)

        hx5 = RSU4F(**network_args)(hx)
        hx = MaxPool2D(**pool_args)(hx5)

        hx6 = RSU4F(**network_args)(hx)
        hx6dup = UpSampling2D(**up_args)(hx6)
        side6_in = Conv2D(**convo2d_args)(hx6)
        side6 = UpSampling2D(size=(32, 32), interpolation='bilinear')(side6_in)

        hx5d = RSU4F(**network_args)(tf.concat([hx6dup, hx5], axis=3))
        hx5dup = UpSampling2D(**up_args)(hx5d)
        side5_in = Conv2D(**convo2d_args)(hx5d)
        side5 = UpSampling2D(size=(16, 16), interpolation='bilinear')(side5_in)

        hx4d = RSU4(**network_args)(tf.concat([hx5dup, hx4], axis=3))
        hx4dup = UpSampling2D(**up_args)(hx4d)
        side4_in = Conv2D(**convo2d_args)(hx4d)
        side4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(side4_in)

        hx3d = RSU5(**network_args)(tf.concat([hx4dup, hx3], axis=3))
        hx3dup = UpSampling2D(**up_args)(hx3d)
        side3_in = Conv2D(**convo2d_args)(hx3d)
        side3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(side3_in)

        hx2d = RSU6(**network_args)(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = UpSampling2D(**up_args)(hx2d)
        side2_in = Conv2D(**convo2d_args)(hx2d)
        side2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(side2_in)
        
        hx1d = RSU7(**network_args)(tf.concat([hx2dup, hx1], axis=3))
        side1 = Conv2D(**convo2d_args)(hx1d)
        outconv = Conv2D(1, (1, 1), padding='same')(tf.concat([side1, side2, side3, side4, side5, side6], axis=3))

        sigmoid = keras.activations.sigmoid
        
        return tf.stack([sigmoid(outconv), sigmoid(side1), sigmoid(side2), sigmoid(side3), sigmoid(side4), sigmoid(side5), sigmoid(side6)])
 
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name='U2NetTF')
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.up_transpose = None
    model.kernel_regularizer = kernel_regularizer
    return model