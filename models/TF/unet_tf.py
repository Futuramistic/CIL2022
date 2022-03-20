from cv2 import normalize
from pandas import concat
import tensorflow as tf
from keras.layers import *

class _Down_Block(tf.keras.Model):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(_Down_Block,self).__init__(self,name=name,**kwargs)
        self.convo_block = _Convo_Block(name+"-convo-block",dropout,filters,kernel_init,normalize,**kwargs)
        self.pool = MaxPool2D(pool_size=(2,2),strides=2,padding='same')

    def call(self,input):
        x1 = self.convo_block(input)
        x2 = self.pool(x1)
        return (x1,x2)

class _Up_Convo(tf.keras.Model):
    def __init__(self,name="up-convo",filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(_Up_Convo,self).__init__(self,name=name,**kwargs)
        self.normalize = normalize
        self.up_convo1 = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_init)
        self.norm1 = BatchNormalization()
    
    def call(self,inputs):
        x = self.up_convo1(inputs)
        if self.normalize:
            x = self.norm1(x)
        return x

class _Up_Block(tf.keras.Model):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(_Up_Block,self).__init__(self,name=name,**kwargs)
        self.up_convo1 = _Up_Convo(name=name+"-up-convo",filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.convo_block1 = _Convo_Block(name+"-convo-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    def call(self,inputs,merger):
        x = self.up_convo1(inputs)
        x = Concatenate(axis=3)([x,merger])
        return self.convo_block1(x)

class _Convo_Block(tf.keras.Model):
    def __init__(self,name="convo-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False, **kwargs):
        super(_Convo_Block, self).__init__(self,name=name, **kwargs)
        
        self.normalize = normalize
        self.convo1 = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=kernel_init)
        self.norm1 = BatchNormalization()
        self.act1 = Activation(activation='relu')
        self.drop1 = Dropout(dropout)
        self.convo2 = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=kernel_init)
        self.norm2 = BatchNormalization()
        self.actv2 = Activation(activation='relu')
        self.drop2 = Dropout(dropout)

    def call(self, inputs):
        x = self.convo1(inputs)
        if self.normalize:
            x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.convo2(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv2(x)
        return self.drop2(x)

class Unet_TF(tf.keras.Model):
    def __init__(self,name="UNET(TF)",dropout=0.5,kernel_init='he_normal',normalize=False,**kwargs):
        super(Unet_TF,self).__init__(self,name=name, **kwargs)

        self.down_block1 = _Down_Block(name+"-down-block-1",dropout,32,kernel_init,normalize)
        self.down_block2 = _Down_Block(name+"-down-block-2",dropout,64,kernel_init,normalize)
        self.down_block3 = _Down_Block(name+"-down-block-3",dropout,128,kernel_init,normalize)
        self.down_block4 = _Down_Block(name+"-down-block-4",dropout,256,kernel_init,normalize)

        self.convo_block = _Convo_Block(name+"-convo-block",dropout,512,kernel_init,normalize)

        self.up_block1 = _Up_Block(name+"-up-block-1",dropout,256,kernel_init,normalize)
        self.up_block2 = _Up_Block(name+"-up-block-2",dropout,128,kernel_init,normalize)
        self.up_block3 = _Up_Block(name+"-up-block-3",dropout,64,kernel_init,normalize)
        self.up_block4 = _Up_Block(name+"-up-block-4",dropout,32,kernel_init,normalize)

        self.final_convo = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
    
    def call(self,inputs):
        # GOING DOWN
        convo1,pool1 = self.down_block1(inputs)
        convo2,pool2 = self.down_block2(pool1)
        convo3,pool3 = self.down_block3(pool2)
        convo4,pool4 = self.down_block4(pool3)

        convo5 = self.convo_block(pool4)

        # GOING UP
        up1 = self.up_block1(convo5,convo4)
        up2 = self.up_block2(up1,convo3)
        up3 = self.up_block3(up2,convo2)
        up4 = self.up_block4(up3,convo1)
        return self.final_convo(up4)

class UnetPlus_TF(tf.keras.Model):
    def __init__(self,name="UNETPlus_TF",dropout=0.5,kernel_init='he_normal',normalize=False, average = False,**kwargs):
        super(UnetPlus_TF,self).__init__(self,name=name, **kwargs)
        self.average = average
        self.nb_filters = [32,64,128,256,512]
        self.down_block1 = _Down_Block(name+"-down-block-1",dropout,self.nb_filters[0],kernel_init,normalize)
        self.down_block2 = _Down_Block(name+"-down-block-2",dropout,self.nb_filters[1],kernel_init,normalize)
        self.down_block3 = _Down_Block(name+"-down-block-3",dropout,self.nb_filters[2],kernel_init,normalize)
        self.down_block4 = _Down_Block(name+"-down-block-4",dropout,self.nb_filters[3],kernel_init,normalize)

        self.convo_block1_2 = _Convo_Block(name+"-convo-block-1_2",dropout,self.nb_filters[0],kernel_init,normalize, **kwargs)
        self.convo_block2_2 = _Convo_Block(name+"-convo-block-2_2",dropout,self.nb_filters[1],kernel_init,normalize, **kwargs)
        self.convo_block1_3 = _Convo_Block(name+"-convo-block-1_3",dropout,self.nb_filters[0],kernel_init,normalize, **kwargs)
        self.convo_block3_2 = _Convo_Block(name+"-convo-block-3_2",dropout,self.nb_filters[2],kernel_init,normalize, **kwargs)
        self.convo_block2_3 = _Convo_Block(name+"-convo-block-2_3",dropout,self.nb_filters[1],kernel_init,normalize, **kwargs)
        self.convo_block1_4 = _Convo_Block(name+"-convo-block-1_4",dropout,self.nb_filters[0],kernel_init,normalize, **kwargs)
        self.convo_block5_1 = _Convo_Block(name+"-convo-block-5_1",dropout,self.nb_filters[4],kernel_init,normalize, **kwargs)
        self.convo_block4_2 = _Convo_Block(name+"-convo-block-4_2",dropout,self.nb_filters[3],kernel_init,normalize, **kwargs)
        self.convo_block3_3 = _Convo_Block(name+"-convo-block-3_3",dropout,self.nb_filters[2],kernel_init,normalize, **kwargs)
        self.convo_block2_4 = _Convo_Block(name+"-convo-block-2_4",dropout,self.nb_filters[1],kernel_init,normalize, **kwargs)
        self.convo_block1_5 = _Convo_Block(name+"-convo-block-1_5",dropout,self.nb_filters[0],kernel_init,normalize, **kwargs)

        self.up_block1_2 = _Up_Convo(name+"-up-convo-2_1",self.nb_filters[0],kernel_init,normalize,**kwargs)
        self.up_block2_2 = _Up_Convo(name+"-up-convo-2_2",self.nb_filters[1],kernel_init,normalize,**kwargs)
        self.up_block1_3 = _Up_Convo(name+"-up-convo-1_3",self.nb_filters[0],kernel_init,normalize,**kwargs)
        self.up_block3_2 = _Up_Convo(name+"-up-convo-3_2",self.nb_filters[2],kernel_init,normalize,**kwargs)
        self.up_block2_3 = _Up_Convo(name+"-up-convo-2_3",self.nb_filters[1],kernel_init,normalize,**kwargs)
        self.up_block1_4 = _Up_Convo(name+"-up-convo-1_4",self.nb_filters[0],kernel_init,normalize,**kwargs)
        self.up_block4_2 = _Up_Convo(name+"-up-convo-4_2",self.nb_filters[3],kernel_init,normalize,**kwargs)
        self.up_block3_3 = _Up_Convo(name+"-up-convo-3_3",self.nb_filters[2],kernel_init,normalize,**kwargs)
        self.up_block2_4 = _Up_Convo(name+"-up-convo-2_4",self.nb_filters[1],kernel_init,normalize,**kwargs)
        self.up_block1_5 = _Up_Convo(name+"-up-convo-1_5",self.nb_filters[0],kernel_init,normalize,**kwargs)

        self.concat1 = Concatenate(axis=3)
        self.concat2 = Concatenate(axis=3)
        self.concat3 = Concatenate(axis=3)
        self.concat4 = Concatenate(axis=3)
        self.concat5 = Concatenate(axis=3)
        self.concat6 = Concatenate(axis=3)
        self.concat7 = Concatenate(axis=3)
        self.concat8 = Concatenate(axis=3)
        self.concat9 = Concatenate(axis=3)
        self.concat10 = Concatenate(axis=3)

        self.output1 = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output2 = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output3 = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.output4 = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)
        self.avr = Average()

    def call(self, inputs):
        convo1_1,pool1 = self.down_block1(inputs)
        convo2_1,pool2 = self.down_block2(pool1)
        
        up1_2 = self.up_block1_2(convo2_1)
        convo1_2 = self.concat1([up1_2,convo1_1])
        convo1_2 = self.convo_block1_2(convo1_2)

        convo3_1,pool3 = self.down_block3(pool2)

        up2_2 = self.up_block2_2(convo3_1)
        convo2_2 = self.concat2([up2_2,convo2_1])
        convo2_2 = self.convo_block2_2(convo2_2)

        up1_3 = self.up_block1_3(convo2_2)
        convo1_3 = self.concat3([up1_3,convo1_1,convo1_2])
        convo1_3 = self.convo_block1_3(convo1_3)

        convo4_1, pool4 = self.down_block4(pool3)

        up3_2 = self.up_block3_2(convo4_1)
        convo3_2 = self.concat4([up3_2,convo3_1])
        convo3_2 = self.convo_block3_2(convo3_2)

        up2_3 = self.up_block2_3(convo3_2)
        convo2_3 = self.concat5([up2_3, convo2_1, convo2_2])
        convo2_3 = self.convo_block2_3(convo2_3)

        up1_4 = self.up_block1_4(convo2_3)
        convo1_4 = self.concat6([up1_4, convo1_1, convo1_2, convo1_3])
        convo1_4 = self.convo_block1_4(convo1_4)

        convo5_1 = self.convo_block5_1(pool4)

        up4_2 = self.up_block4_2(convo5_1)
        convo4_2 = self.concat7([up4_2, convo4_1])
        convo4_2 = self.convo_block4_2(up4_2)

        up3_3 = self.up_block3_3(convo4_2)
        convo3_3 = self.concat8([up3_3, convo3_1, convo3_2])
        convo3_3 = self.convo_block3_3(convo3_3)

        up2_4 = self.up_block2_4(convo3_3)
        convo2_4 = self.concat9([up2_4, convo2_1, convo2_2, convo2_3]) 
        convo2_4 = self.convo_block2_4(convo2_4)

        up1_5 = self.up_block1_5(convo2_4)
        convo1_5 = self.concat10([up1_5, convo1_1, convo1_2, convo1_3, convo1_4])
        convo1_5 = self.convo_block1_5(convo1_5)

        output1 = self.output1(convo1_2)
        output2 = self.output2(convo1_3)
        output3 = self.output3(convo1_4)
        output4 = self.output4(convo1_5)

        if self.average:
            return self.avr([output1,output2,output3,output4])
        return output4

class _Attention_Block(tf.keras.Model):
    def __init__(self,name="Attention-block",filters=64,normalize=False,**kwargs):
        super(_Attention_Block,self).__init__(self,name=name,**kwargs)
        self.normalize = normalize
        self.theta_x = Conv2D(filters=filters, kernel_size=1, strides=(1,1), padding='same')
        self.norm1 = BatchNormalization()
        self.phi_g = Conv2D(filters=filters, kernel_size=1, strides=(1,1), padding='same')
        self.norm2 = BatchNormalization()
        self.add = Add()
        self.f = Activation(activation='relu')
        self.psi_f = Conv2D(filters=1,kernel_size=1,strides=(1,1),padding='same')
        self.norm3 = BatchNormalization()
        self.activ1 = Activation(activation='sigmoid')
        self.activ2 = Activation(activation='sigmoid')
        self.att_x = Multiply()

    def call(self,x,g):
        theta_x = self.theta_x(x)
        if self.normalize:
            theta_x = self.norm1
        phi_g = self.phi_g(g)
        if self.normalize:
            phi_g = self.norm2
        add = self.add([phi_g,theta_x])
        f = self.activ1(add)
        psi_f = self.psi_f(f)
        if self.normalize:
            psi_f = self.norm3
        rate = self.activ2(psi_f)
        return self.att_x([x,rate])

class _Attention_Block_Up(tf.keras.Model):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(_Attention_Block_Up,self).__init__(self,name=name,**kwargs)
        self.up = _Up_Convo(name=name+"-up-conv",filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.att = _Attention_Block(name=name+"-attention-block",filters=filters,normalize=normalize)
        self.concat = Concatenate(axis=3)
        self.convo = _Convo_Block(name=name+"-convo-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    def call(self,down_layer,input):
        up = self.up(down_layer)
        att = self.att(x=input,g=up)
        concat = self.concat([up,att])
        return self.convo(concat)

class Att_Unet_TF(tf.keras.Model):
    def __init__(self,name="Attention_Unet(TF)",dropout=0.5,kernel_init='he_normal',normalize=False,**kwargs):
        super(Att_Unet_TF,self).__init__(self,name=name,**kwargs)
        self.nb_filters = [32,64,128,256,512]

        self.down_block1 = _Down_Block(name+"-down-block-1",dropout,self.nb_filters[0],kernel_init,normalize)
        self.down_block2 = _Down_Block(name+"-down-block-2",dropout,self.nb_filters[1],kernel_init,normalize)
        self.down_block3 = _Down_Block(name+"-down-block-3",dropout,self.nb_filters[2],kernel_init,normalize)
        self.down_block4 = _Down_Block(name+"-down-block-4",dropout,self.nb_filters[3],kernel_init,normalize)

        self.convo_block = _Convo_Block(name+"-convo-block",dropout,self.nb_filters[4],kernel_init,normalize)

        self.up_block1 = _Attention_Block_Up(name+"-att-block-1",dropout,self.nb_filters[3],kernel_init,normalize)
        self.up_block2 = _Attention_Block_Up(name+"-att-block-2",dropout,self.nb_filters[2],kernel_init,normalize)
        self.up_block3 = _Attention_Block_Up(name+"-att-block-3",dropout,self.nb_filters[1],kernel_init,normalize)
        self.up_block4 = _Attention_Block_Up(name+"-att-block-4",dropout,self.nb_filters[0],kernel_init,normalize)

        self.final_convo = Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=kernel_init)

    def call(self,inputs):
        # GOING DOWN
        convo1,pool1 = self.down_block1(inputs)
        convo2,pool2 = self.down_block2(pool1)
        convo3,pool3 = self.down_block3(pool2)
        convo4,pool4 = self.down_block4(pool3)

        convo5 = self.convo_block(pool4)

        # GOING UP
        up1 = self.up_block1(convo5,convo4)
        up2 = self.up_block2(up1,convo3)
        up3 = self.up_block3(up2,convo2)
        up4 = self.up_block4(up3,convo1)
        return self.final_convo(up4)