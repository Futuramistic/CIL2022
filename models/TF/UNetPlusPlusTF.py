import tensorflow as tf
from keras.layers import *
import tensorflow.keras as K
from .blocks import *

def UnetPlusPlusTF(input_shape,name="UNetTF",dropout=0.5,kernel_init='he_normal',normalize=True, up_transpose=True, average=False):

    def __build_model(inputs):
            nb_filters = [32,64,128,256,512]
            if up_transpose:
                up_block = Transpose_Block
            else:
                up_block = UpSampleConvo_Block

            out_args = {
                'filters': 1,
                'kernel_size':(1,1),
                'padding':'same',
                'activation':'sigmoid',
                'kernel_initializer':kernel_init
            }

            convo1_1,pool1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(inputs)
            convo2_1,pool2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(pool1)
            
            up1_2 = up_block(name=name+"-up-convo-2_1",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo2_1)
            convo1_2 = Concatenate(name=name+"-concat1",axis=3)([up1_2,convo1_1])
            convo1_2 = Convo_Block(name=name+"-convo-block-1_2",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo1_2)
            
            convo3_1,pool3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(pool2)
            
            up2_2 = up_block(name=name+"-up-convo-2_2",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo3_1)
            convo2_2 = Concatenate(name=name+"-concat2",axis=3)([up2_2,convo2_1])
            convo2_2 = Convo_Block(name=name+"-convo-block-2_2",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo2_2)

            up1_3 = up_block(name=name+"-up-convo-1_3",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo2_2)
            convo1_3 = Concatenate(name=name+"-concat3",axis=3)([up1_3,convo1_1,convo1_2])
            convo1_3 = Convo_Block(name=name+"-convo-block-1_3",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo1_3)

            convo4_1, pool4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=nb_filters[3],kernel_init=kernel_init,normalize=normalize)(pool3)

            up3_2 = up_block(name=name+"-up-convo-3_2",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(convo4_1)
            convo3_2 = Concatenate(name=name+"-concat4",axis=3)([up3_2,convo3_1])
            convo3_2 = Convo_Block(name=name+"-convo-block-3_2",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(convo3_2)

            up2_3 = up_block(name=name+"-up-convo-2_3",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo3_2)
            convo2_3 = Concatenate(name=name+"-concat5",axis=3)([up2_3, convo2_1, convo2_2])
            convo2_3 = Convo_Block(name=name+"-convo-block-2_3",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo2_3)

            up1_4 = up_block(name=name+"-up-convo-1_4",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo2_3)
            convo1_4 = Concatenate(name=name+"-concat6",axis=3)([up1_4, convo1_1, convo1_2, convo1_3])
            convo1_4 = Convo_Block(name=name+"-convo-block-1_4",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo1_4)

            convo5_1 = Convo_Block(name=name+"-convo-block-5_1",dropout=dropout,filters=nb_filters[4],kernel_init=kernel_init,normalize=normalize)(pool4)

            up4_2 = up_block(name=name+"-up-convo-4_2",dropout=dropout,filters=nb_filters[3],kernel_init=kernel_init,normalize=normalize)(convo5_1)
            convo4_2 = Concatenate(name=name+"-concat7",axis=3)([up4_2, convo4_1])
            convo4_2 = Convo_Block(name=name+"-convo-block-4_2",dropout=dropout,filters=nb_filters[3],kernel_init=kernel_init,normalize=normalize)(convo4_2)

            up3_3 = up_block(name=name+"-up-convo-3_3",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(convo4_2)
            convo3_3 =  Concatenate(name=name+"-concat8",axis=3)([up3_3, convo3_1, convo3_2])
            convo3_3 = Convo_Block(name=name+"-convo-block-3_3",dropout=dropout,filters=nb_filters[2],kernel_init=kernel_init,normalize=normalize)(convo3_3)

            up2_4 = up_block(name=name+"-up-convo-2_4",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo3_3)
            convo2_4 = Concatenate(name=name+"-concat9",axis=3)([up2_4, convo2_1, convo2_2, convo2_3]) 
            convo2_4 = Convo_Block(name=name+"-convo-block-2_4",dropout=dropout,filters=nb_filters[1],kernel_init=kernel_init,normalize=normalize)(convo2_4)

            up1_5 = up_block(name=name+"-up-convo-1_5",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo2_4)
            convo1_5 = Concatenate(name=name+"-concat10",axis=3)([up1_5, convo1_1, convo1_2, convo1_3, convo1_4])
            convo1_5 = Convo_Block(name=name+"-convo-block-1_5",dropout=dropout,filters=nb_filters[0],kernel_init=kernel_init,normalize=normalize)(convo1_5)

            output1 = Conv2D(name=name+"-output-1",**out_args)(convo1_2)
            output2 = Conv2D(name=name+"-output-2",**out_args)(convo1_3)
            output3 = Conv2D(name=name+"-output-3",**out_args)(convo1_4)
            output4 = Conv2D(name=name+"-output-4",**out_args)(convo1_5)
            
            if average:
                return Average(name=name+"-final-average")([output1,output2,output3,output4])
            return output4

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model