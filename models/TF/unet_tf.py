import tensorflow as tf
from keras.layers import *
from blocks import *

class Unet_TF(tf.keras.Model):
    def __init__(self,name="Unet-TF-",dropout=0.5,kernel_init='he_normal',normalize=False, up_transpose=True,**kwargs):
        super(Unet_TF,self).__init__(name=name, **kwargs)
        self.nb_filters = [32,64,128,256,512]
        if up_transpose:
            self.up_block = Transpose_Block
        else:
            self.up_block = UpSampleConvo_Block
        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)

        self.up_block1 = Up_Block(name=name+"-up-block-1",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block2 = Up_Block(name=name+"-up-block-2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block3 = Up_Block(name=name+"-up-block-3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block4 = Up_Block(name=name+"-up-block-4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)

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

class UnetPlusPlus_TF(tf.keras.Model):
    def __init__(self,name="UNetPlusPlus-TF-",dropout=0.5,kernel_init='he_normal',normalize=False, deep_supervision = False, up_transpose = True,**kwargs):
        super(UnetPlusPlus_TF,self).__init__(name=name, **kwargs)
        self.average = average
        self.nb_filters = [32,64,128,256,512]

        if up_transpose:
            self.up_block = Transpose_Block
        else:
            self.up_block = UpSampleConvo_Block

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

        self.up_block1_2 = self.up_block(name=name+"-up-convo-2_1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_2 = self.up_block(name=name+"-up-convo-2_2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_3 = self.up_block(name=name+"-up-convo-1_3",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block3_2 = self.up_block(name=name+"-up-convo-3_2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_3 = self.up_block(name=name+"-up-convo-2_3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_4 = self.up_block(name=name+"-up-convo-1_4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.up_block4_2 = self.up_block(name=name+"-up-convo-4_2",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)
        self.up_block3_3 = self.up_block(name=name+"-up-convo-3_3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.up_block2_4 = self.up_block(name=name+"-up-convo-2_4",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.up_block1_5 = self.up_block(name=name+"-up-convo-1_5",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.concat = Concatenate(name=name+"-concat1",axis=3)

        self.output1 = Conv2D(name=name+"-output-1",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output2 = Conv2D(name=name+"-output-2",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output3 = Conv2D(name=name+"-output-3",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output4 = Conv2D(name=name+"-output-4",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.avr = Average(name=name+"-final-average")

    def call(self, inputs, training=None, **kwargs):
        convo1_1,pool1 = self.down_block1(inputs,training=training, **kwargs)
        convo2_1,pool2 = self.down_block2(pool1,training=training, **kwargs)
        
        up1_2 = self.up_block1_2(convo2_1,training=training, **kwargs)
        convo1_2 = self.concat([up1_2,convo1_1])
        convo1_2 = self.convo_block1_2(convo1_2,training=training, **kwargs)

        convo3_1,pool3 = self.down_block3(pool2,training=training, **kwargs)

        up2_2 = self.up_block2_2(convo3_1,training=training, **kwargs)
        convo2_2 = self.concat([up2_2,convo2_1])
        convo2_2 = self.convo_block2_2(convo2_2,training=training, **kwargs)

        up1_3 = self.up_block1_3(convo2_2,training=training, **kwargs)
        convo1_3 = self.concat([up1_3,convo1_1,convo1_2])
        convo1_3 = self.convo_block1_3(convo1_3,training=training, **kwargs)

        convo4_1, pool4 = self.down_block4(pool3,training=training, **kwargs)

        up3_2 = self.up_block3_2(convo4_1,training=training, **kwargs)
        convo3_2 = self.concat([up3_2,convo3_1])
        convo3_2 = self.convo_block3_2(convo3_2,training=training, **kwargs)

        up2_3 = self.up_block2_3(convo3_2,training=training, **kwargs)
        convo2_3 = self.concat([up2_3, convo2_1, convo2_2])
        convo2_3 = self.convo_block2_3(convo2_3,training=training, **kwargs)

        up1_4 = self.up_block1_4(convo2_3,training=training, **kwargs)
        convo1_4 = self.concat([up1_4, convo1_1, convo1_2, convo1_3])
        convo1_4 = self.convo_block1_4(convo1_4,training=training, **kwargs)

        convo5_1 = self.convo_block5_1(pool4,training=training, **kwargs)

        up4_2 = self.up_block4_2(convo5_1,training=training, **kwargs)
        convo4_2 = self.concat([up4_2, convo4_1])
        convo4_2 = self.convo_block4_2(up4_2,training=training, **kwargs)

        up3_3 = self.up_block3_3(convo4_2,training=training, **kwargs)
        convo3_3 = self.concat([up3_3, convo3_1, convo3_2])
        convo3_3 = self.convo_block3_3(convo3_3,training=training, **kwargs)

        up2_4 = self.up_block2_4(convo3_3,training=training, **kwargs)
        convo2_4 = self.concat([up2_4, convo2_1, convo2_2, convo2_3]) 
        convo2_4 = self.convo_block2_4(convo2_4,training=training, **kwargs)

        up1_5 = self.up_block1_5(convo2_4,training=training, **kwargs)
        convo1_5 = self.concat([up1_5, convo1_1, convo1_2, convo1_3, convo1_4])
        convo1_5 = self.convo_block1_5(convo1_5,training=training, **kwargs)

        output1 = self.output1(convo1_2,**kwargs)
        output2 = self.output2(convo1_3,**kwargs)
        output3 = self.output3(convo1_4,**kwargs)
        output4 = self.output4(convo1_5,**kwargs)

        if self.average:
            return self.avr([output1,output2,output3,output4])
        return output4

class Att_Unet_TF(tf.keras.Model):
    def __init__(self,name="Att_Unet-TF-",dropout=0.5,kernel_init='he_normal',normalize=False,up_transpose = True,**kwargs):
        super(Att_Unet_TF,self).__init__(name=name,**kwargs)
        self.nb_filters = [32,64,128,256,512]

        if up_transpose:
            self.up_block = Transpose_Block
        else:
            self.up_block = UpSampleConvo_Block

        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)

        self.up_block1 = Attention_Block_Concat(name=name+"-att-block-1",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block2 = Attention_Block_Concat(name=name+"-att-block-2",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block3 = Attention_Block_Concat(name=name+"-att-block-3",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)
        self.up_block4 = Attention_Block_Concat(name=name+"-att-block-4",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,up_convo=self.up_block)

        self.final_convo = Conv2D(name=name+"-final-convo",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)

    def call(self, inputs, training=None, **kwargs):
        # GOING DOWN
        convo1,pool1 = self.down_block1(inputs,training=training, **kwargs)
        convo2,pool2 = self.down_block2(pool1,training=training, **kwargs)
        convo3,pool3 = self.down_block3(pool2,training=training, **kwargs)
        convo4,pool4 = self.down_block4(pool3,training=training, **kwargs)

        convo5 = self.convo_block(pool4,training=training, **kwargs)

        # GOING UP
        up1 = self.up_block1(g=convo5,x=convo4,training=training, **kwargs)
        up2 = self.up_block2(g=up1,x=convo3,training=training, **kwargs)
        up3 = self.up_block3(g=up2,x=convo2,training=training, **kwargs)
        up4 = self.up_block4(g=up3,x=convo1,training=training, **kwargs)
        return self.final_convo(up4, **kwargs)

class Att_UnetPlusPlus_TF(tf.keras.Model):
    def __init__(self,name="Att_UnetPlusPlus-TF-",dropout=0.5,kernel_init='he_normal',normalize=False,deep_supervision=False, up_transpose = True,**kwargs):
        super(Att_UnetPlusPlus_TF,self).__init__(name=name,**kwargs)
        self.deep_supervision = deep_supervision
        self.nb_filters = [32,64,128,256,512]

        if up_transpose:
            self.up_block = Transpose_Block
        else:
            self.up_block = UpSampleConvo_Block

        self.down_block1 = Down_Block(name=name+"-down-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)
        self.down_block2 = Down_Block(name=name+"-down-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)
        self.down_block3 = Down_Block(name=name+"-down-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)
        self.down_block4 = Down_Block(name=name+"-down-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.convo_block_bottom = Convo_Block(name=name+"-convo-block-bottom",dropout=dropout,filters=self.nb_filters[4],kernel_init=kernel_init,normalize=normalize)
        
        self.att_block1 = Attention_Block(name=name+"-att-block-1",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block1 = self.up_block(name=name+"-up-convo-block-1",filters=self.nb_filters[0], kernel_init=kernel_init,normalize=normalize)
        self.convo_block1 = Convo_Block(name=name+"-convo-block-1",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.att_block2 = Attention_Block(name=name+"-att-block-2",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block2 = self.up_block(name=name+"-up-convo-block-2",filters=self.nb_filters[1], kernel_init=kernel_init,normalize=normalize)
        self.convo_block2 = Convo_Block(name=name+"-convo-block-2",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)

        self.att_block3 = Attention_Block(name=name+"-att-block-3",filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block3 = self.up_block(name=name+"-up-convo-block-3",filters=self.nb_filters[2], kernel_init=kernel_init,normalize=normalize)
        self.convo_block3 = Convo_Block(name=name+"-convo-block-3",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)

        self.att_block4 = Attention_Block(name=name+"-att-block-4",filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block4 = self.up_block(name=name+"-up-convo-block-4",filters=self.nb_filters[3], kernel_init=kernel_init,normalize=normalize)
        self.convo_block4 = Convo_Block(name=name+"-convo-block-4",dropout=dropout,filters=self.nb_filters[3],kernel_init=kernel_init,normalize=normalize)

        self.att_block5 = Attention_Block(name=name+"-att-block-5",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)      
        self.up_convo_block5 = self.up_block(name=name+"-up-convo-block-5",filters=self.nb_filters[0], kernel_init=kernel_init,normalize=normalize)
        self.convo_block5 = Convo_Block(name=name+"-convo-block-5",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.att_block6 = Attention_Block(name=name+"-att-block-6",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block6 = self.up_block(name=name+"-up-convo-block-6",filters=self.nb_filters[1], kernel_init=kernel_init,normalize=normalize)
        self.convo_block6 = Convo_Block(name=name+"-convo-block-6",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)

        self.att_block7 = Attention_Block(name=name+"-att-block-7",filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block7 = self.up_block(name=name+"-up-convo-block-7",filters=self.nb_filters[2], kernel_init=kernel_init,normalize=normalize)     
        self.convo_block7 = Convo_Block(name=name+"-convo-block-7",dropout=dropout,filters=self.nb_filters[2],kernel_init=kernel_init,normalize=normalize)

        self.att_block8 = Attention_Block(name=name+"-att-block-8",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block8 = self.up_block(name=name+"-up-convo-block-8",filters=self.nb_filters[0], kernel_init=kernel_init,normalize=normalize)
        self.convo_block8 = Convo_Block(name=name+"-convo-block-8",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.att_block9 = Attention_Block(name=name+"-att-block-9",filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block9 = self.up_block(name=name+"-up-convo-block-9",filters=self.nb_filters[1], kernel_init=kernel_init,normalize=normalize)
        self.convo_block9 = Convo_Block(name=name+"-convo-block-9",dropout=dropout,filters=self.nb_filters[1],kernel_init=kernel_init,normalize=normalize)

        self.att_block10 = Attention_Block(name=name+"-att-block-10",filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize,dropout=dropout,up_convo=self.up_block)
        self.up_convo_block10 = self.up_block(name=name+"-up-convo-block-10",filters=self.nb_filters[0], kernel_init=kernel_init,normalize=normalize)
        self.convo_block10 = Convo_Block(name=name+"-convo-block-10",dropout=dropout,filters=self.nb_filters[0],kernel_init=kernel_init,normalize=normalize)

        self.concat = Concatenate(name=name+"-concat",axis=3)
        self.max_pool = MaxPooling2D(pool_size=(2,2),strides=2,padding='same',name=name+"-max-pool")
        self.average = Average(name=name+"-average")

        self.output1 = Conv2D(name=name+"-output-1",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output2 = Conv2D(name=name+"-output-2",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output3 = Conv2D(name=name+"-output-3",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output4 = Conv2D(name=name+"-output-4",filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)

    def call(self, inputs, training=None, **kwargs):
        x0_0,pool1 = self.down_block1(inputs,training=training, **kwargs)
        x1_0,pool2 = self.down_block2(pool1,training=training, **kwargs)
        x2_0,pool3 = self.down_block3(pool2,training=training, **kwargs)
        x3_0,pool4 = self.down_block4(pool3,training=training, **kwargs)

        x4_0 = self.convo_block_bottom(pool4,training=training, **kwargs)
        
        att_x00 = self.att_block1(x=x0_0,g=x1_0,training=training, **kwargs)
        up_x10 = self.up_convo_block1(x1_0,training=training, **kwargs)
        x0_1 = self.convo_block1(self.concat([att_x00,up_x10]),training=training, **kwargs)

        att_x10 = self.att_block2(x=x1_0,g=x2_0,training=training, **kwargs)
        up_x20 = self.up_convo_block2(x2_0,training=training, **kwargs)
        down_x01 = self.max_pool(x0_1)
        x1_1 = self.convo_block2(self.concat([att_x10,up_x20,down_x01]),training=training, **kwargs)

        att_x20 = self.att_block3(x=x2_0,g=x3_0,training=training, **kwargs)
        up_x30 = self.up_convo_block3(x3_0,training=training, **kwargs)
        down_x11 = self.max_pool(x1_1)
        x2_1 = self.convo_block3(self.concat([att_x20,up_x30,down_x11]),training=training, **kwargs)

        att_x30 = self.att_block4(x=x3_0, g=x4_0,training=training, **kwargs)
        up_x40 = self.up_convo_block4(x4_0,training=training, **kwargs)
        down_x21 = self.max_pool(x2_1)
        x3_1 = self.convo_block4(self.concat([att_x30,up_x40,down_x21]),training=training, **kwargs)

        att_x01 = self.att_block5(x=x0_1,g=x1_1,training=training, **kwargs)
        up_x11 = self.up_convo_block5(x1_1,training=training, **kwargs)
        x0_2 = self.convo_block5(self.concat([att_x00,att_x01,up_x11]),training=training, **kwargs)

        att_x11 = self.att_block6(x=x1_1,g=x2_1,training=training, **kwargs)
        up_x21 = self.up_convo_block6(x2_1,training=training, **kwargs)
        down_x02 = self.max_pool(x0_2)
        x1_2 = self.convo_block6(self.concat([att_x10,att_x11,up_x21,down_x02]),training=training, **kwargs)

        att_x21 = self.att_block7(x=x2_1,g=x3_1,training=training, **kwargs)
        up_x31 = self.up_convo_block7(x3_1,training=training, **kwargs)
        down_x12 = self.max_pool(x1_2)
        x2_2 = self.convo_block7(self.concat([att_x20,att_x21,up_x31,down_x12]),training=training, **kwargs)

        att_x02 = self.att_block8(x=x0_2,g=x1_2,training=training, **kwargs)
        up_x12 = self.up_convo_block8(x1_2,training=training, **kwargs)
        x0_3 = self.convo_block8(self.concat([att_x00,att_x01,att_x02,up_x12]),training=training, **kwargs)

        att_x12 = self.att_block9(x=x1_2,g=x2_2,training=training, **kwargs)
        up_x22 = self.up_convo_block9(x2_2,training=training, **kwargs)
        down_x03 = self.max_pool(x0_3)
        x1_3 = self.convo_block9(self.concat([att_x10,att_x11,att_x12,up_x22,down_x03]),training=training, **kwargs)

        att_x03 = self.att_block10(x=x0_3,g=x1_3,training=training, **kwargs)
        up_x13 = self.up_convo_block10(x1_3,training=training, **kwargs)
        x0_4 = self.convo_block10(self.concat([att_x00,att_x01,att_x02,att_x03,up_x13]),training=training, **kwargs)

        output1 = self.output1(x0_1,training=training, **kwargs)
        output2 = self.output2(x0_2,training=training, **kwargs)
        output3 = self.output3(x0_3,training=training, **kwargs)
        output4 = self.output4(x0_4,training=training, **kwargs)

        if self.deep_supervision:
            return self.average([output1,output2,output3,output4])
        else:
            return output4

# To check if model compiles try:
# model = UnetPlusPlus_TF(normalize=True,up_transpose=True)
# model.build((32,400,400,3))
# model.summary()