import tensorflow as tf
from keras.layers import *
import tensorflow.keras as K
from .blocks import *

def UNetTF(input_shape,name="UNetTF",dropout=0.5,kernel_init='he_normal',normalize=True, up_transpose=True):

    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        convo1,pool1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(inputs)
        convo2,pool2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(pool1)
        convo3,pool3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(pool2)
        convo4,pool4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=nb_filters[3],kernel_init=kernel_init,normalize=normalize)(pool3)

        convo5 = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=nb_filters[4],kernel_init=kernel_init,normalize=normalize)(pool4)

        up1 = Up_Block(name=name+"-up-block-1",dropout=dropout,filters=nb_filters[3],kernel_init=kernel_init,normalize=normalize,up_convo=up_block)(convo5,convo4)
        up2 = Up_Block(name=name+"-up-block-2",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize,up_convo=up_block)(up1,convo3)
        up3 = Up_Block(name=name+"-up-block-3",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize,up_convo=up_block)(up2,convo2)
        up4 = Up_Block(name=name+"-up-block-4",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize,up_convo=up_block)(up3,convo1)

        return Conv2D(name=name+"-final-convo",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2())(up4)
    
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model