from cv2 import normalize
from pandas import concat
import tensorflow as tf
from keras.layers import *

from blocks import *

class Unet_TF(tf.keras.Model):
    def __init__(self,name="Unet-TF-",dropout=0.5,kernel_init='he_normal',normalize=False,**kwargs):
        super(Unet_TF,self).__init__(name=name, **kwargs)
        self.nb_filters = [32,64,128,256,512]
        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)

        self.up_block1 = Up_Block(name=name+"-up-block-1",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)
        self.up_block2 = Up_Block(name=name+"-up-block-2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block3 = Up_Block(name=name+"-up-block-3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block4 = Up_Block(name=name+"-up-block-4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.final_convo = Conv2D(name=name+"-final-convo",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
    
    def call(self, inputs, training=None, **kwargs):
        # GOING DOWN
        convo1,pool1 = self.down_block1(inputs,training=training, **kwargs)
        convo2,pool2 = self.down_block2(pool1,training=training, **kwargs)
        convo3,pool3 = self.down_block3(pool2,training=training, **kwargs)
        convo4,pool4 = self.down_block4(pool3,training=training, **kwargs)

        convo5 = self.convo_block(pool4,training=training, **kwargs)

        # GOING UP
        up1 = self.up_block1(convo5,convo4,training=training, **kwargs)
        up2 = self.up_block2(up1,convo3,training=training, **kwargs)
        up3 = self.up_block3(up2,convo2,training=training, **kwargs)
        up4 = self.up_block4(up3,convo1,training=training, **kwargs)
        return self.final_convo(up4,training=training, **kwargs)

class UnetPlus_TF(tf.keras.Model):
    def __init__(self,name="UNetPlusPlus-TF-",dropout=0.5,kernel_init='he_normal',normalize=False, average = False,**kwargs):
        super(UnetPlus_TF,self).__init__(name=name, **kwargs)
        self.average = average
        self.nb_filters = [32,64,128,256,512]
        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block1_2 = Convo_Block(name=name+"-convo-block-1_2",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.convo_block2_2 = Convo_Block(name=name+"-convo-block-2_2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.convo_block1_3 = Convo_Block(name=name+"-convo-block-1_3",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.convo_block3_2 = Convo_Block(name=name+"-convo-block-3_2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.convo_block2_3 = Convo_Block(name=name+"-convo-block-2_3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.convo_block1_4 = Convo_Block(name=name+"-convo-block-1_4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.convo_block5_1 = Convo_Block(name=name+"-convo-block-5_1",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)
        self.convo_block4_2 = Convo_Block(name=name+"-convo-block-4_2",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)
        self.convo_block3_3 = Convo_Block(name=name+"-convo-block-3_3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.convo_block2_4 = Convo_Block(name=name+"-convo-block-2_4",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.convo_block1_5 = Convo_Block(name=name+"-convo-block-1_5",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.up_block1_2 = Up_Convo_Block(name=name+"-up-convo-2_1",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_2 = Up_Convo_Block(name=name+"-up-convo-2_2",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_3 = Up_Convo_Block(name=name+"-up-convo-1_3",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block3_2 = Up_Convo_Block(name=name+"-up-convo-3_2",filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_3 = Up_Convo_Block(name=name+"-up-convo-2_3",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_4 = Up_Convo_Block(name=name+"-up-convo-1_4",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block4_2 = Up_Convo_Block(name=name+"-up-convo-4_2",filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)
        self.up_block3_3 = Up_Convo_Block(name=name+"-up-convo-3_3",filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_4 = Up_Convo_Block(name=name+"-up-convo-2_4",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_5 = Up_Convo_Block(name=name+"-up-convo-1_5",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.concat1 = Concatenate(name=name+"-concat1",axis=3)
        self.concat2 = Concatenate(name=name+"-concat2",axis=3)
        self.concat3 = Concatenate(name=name+"-concat3",axis=3)
        self.concat4 = Concatenate(name=name+"-concat4",axis=3)
        self.concat5 = Concatenate(name=name+"-concat5",axis=3)
        self.concat6 = Concatenate(name=name+"-concat6",axis=3)
        self.concat7 = Concatenate(name=name+"-concat7",axis=3)
        self.concat8 = Concatenate(name=name+"-concat8",axis=3)
        self.concat9 = Concatenate(name=name+"-concat9",axis=3)
        self.concat10 = Concatenate(name=name+"-concat10",axis=3)

        self.output1 = Conv2D(name=name+"-output-1",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output2 = Conv2D(name=name+"-output-2",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output3 = Conv2D(name=name+"-output-3",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output4 = Conv2D(name=name+"-output-4",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.avr = Average(name=name+"-final-average")

    def call(self, inputs, training=None, **kwargs):
        convo1_1,pool1 = self.down_block1(inputs,training=training, **kwargs)
        convo2_1,pool2 = self.down_block2(pool1,training=training, **kwargs)
        
        up1_2 = self.up_block1_2(convo2_1,training=training, **kwargs)
        convo1_2 = self.concat1([up1_2,convo1_1])
        convo1_2 = self.convo_block1_2(convo1_2,training=training, **kwargs)

        convo3_1,pool3 = self.down_block3(pool2,training=training, **kwargs)

        up2_2 = self.up_block2_2(convo3_1,training=training, **kwargs)
        convo2_2 = self.concat2([up2_2,convo2_1])
        convo2_2 = self.convo_block2_2(convo2_2,training=training, **kwargs)

        up1_3 = self.up_block1_3(convo2_2,training=training, **kwargs)
        convo1_3 = self.concat3([up1_3,convo1_1,convo1_2])
        convo1_3 = self.convo_block1_3(convo1_3,training=training, **kwargs)

        convo4_1, pool4 = self.down_block4(pool3,training=training, **kwargs)

        up3_2 = self.up_block3_2(convo4_1,training=training, **kwargs)
        convo3_2 = self.concat4([up3_2,convo3_1])
        convo3_2 = self.convo_block3_2(convo3_2,training=training, **kwargs)

        up2_3 = self.up_block2_3(convo3_2,training=training, **kwargs)
        convo2_3 = self.concat5([up2_3, convo2_1, convo2_2])
        convo2_3 = self.convo_block2_3(convo2_3,training=training, **kwargs)

        up1_4 = self.up_block1_4(convo2_3,training=training, **kwargs)
        convo1_4 = self.concat6([up1_4, convo1_1, convo1_2, convo1_3])
        convo1_4 = self.convo_block1_4(convo1_4,training=training, **kwargs)

        convo5_1 = self.convo_block5_1(pool4,training=training, **kwargs)

        up4_2 = self.up_block4_2(convo5_1,training=training, **kwargs)
        convo4_2 = self.concat7([up4_2, convo4_1])
        convo4_2 = self.convo_block4_2(up4_2,training=training, **kwargs)

        up3_3 = self.up_block3_3(convo4_2,training=training, **kwargs)
        convo3_3 = self.concat8([up3_3, convo3_1, convo3_2])
        convo3_3 = self.convo_block3_3(convo3_3,training=training, **kwargs)

        up2_4 = self.up_block2_4(convo3_3,training=training, **kwargs)
        convo2_4 = self.concat9([up2_4, convo2_1, convo2_2, convo2_3]) 
        convo2_4 = self.convo_block2_4(convo2_4,training=training, **kwargs)

        up1_5 = self.up_block1_5(convo2_4,training=training, **kwargs)
        convo1_5 = self.concat10([up1_5, convo1_1, convo1_2, convo1_3, convo1_4])
        convo1_5 = self.convo_block1_5(convo1_5,training=training, **kwargs)

        output1 = self.output1(convo1_2,**kwargs)
        output2 = self.output2(convo1_3,**kwargs)
        output3 = self.output3(convo1_4,**kwargs)
        output4 = self.output4(convo1_5,**kwargs)

        if self.average:
            return self.avr([output1,output2,output3,output4])
        return output4

class Att_Unet_TF(tf.keras.Model):
    def __init__(self,name="Att_Unet-TF-",dropout=0.5,kernel_init='he_normal',normalize=False,**kwargs):
        super(Att_Unet_TF,self).__init__(name=name,**kwargs)
        self.nb_filters = [32,64,128,256,512]

        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)

        self.up_block1 = Attention_Block_Up(name=name+"-att-block-1",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)
        self.up_block2 = Attention_Block_Up(name=name+"-att-block-2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block3 = Attention_Block_Up(name=name+"-att-block-3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block4 = Attention_Block_Up(name=name+"-att-block-4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.final_convo = Conv2D(name=name+"-final-convo",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)

    def call(self, inputs, training=None, **kwargs):
        # GOING DOWN
        convo1,pool1 = self.down_block1(inputs,training=training, **kwargs)
        convo2,pool2 = self.down_block2(pool1,training=training, **kwargs)
        convo3,pool3 = self.down_block3(pool2,training=training, **kwargs)
        convo4,pool4 = self.down_block4(pool3,training=training, **kwargs)

        convo5 = self.convo_block(pool4,training=training, **kwargs)

        # GOING UP
        up1 = self.up_block1(convo5,convo4,training=training, **kwargs)
        up2 = self.up_block2(up1,convo3,training=training, **kwargs)
        up3 = self.up_block3(up2,convo2,training=training, **kwargs)
        up4 = self.up_block4(up3,convo1,training=training, **kwargs)
        return self.final_convo(up4, **kwargs)