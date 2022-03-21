import tensorflow as tf
from keras.layers import *

class Convo_Block(tf.keras.layers.Layer):
    def __init__(self,name="convo-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Convo_Block, self).__init__(name=name,**kwargs)     
        self.normalize = normalize
        self.convo1 = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=kernel_init,name=name+"-conv2D-1")
        self.norm1 = BatchNormalization(name=name+"-batchNorm-1")
        self.actv1 = Activation(activation='relu',name=name+"-activ-1")
        self.drop1 = Dropout(dropout,name=name+"-drop-1")
        self.convo2 = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=kernel_init,name=name+"-conv2D-2")
        self.norm2 = BatchNormalization(name=name+"-batchNorm-2")
        self.actv2 = Activation(activation='relu',name=name+"-activ-2")
        self.drop2 = Dropout(dropout,name=name+"-drop-2")

    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, training=None, **kwargs):
        x = self.convo1(inputs)
        if self.normalize:
            x = self.norm1(x,training=training)
        x = self.actv1(x)
        if training:
            x = self.drop1(x)
        x = self.convo2(x)
        if self.normalize:
            x = self.norm2(x,training=training)
        x = self.actv2(x)
        if training:
            x = self.drop2(x)
        return x

class Down_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Down_Block,self).__init__(name=name,**kwargs)
        self.convo_block = Convo_Block(name+"-convo-block",dropout,filters,kernel_init,normalize)
        self.pool = MaxPool2D(pool_size=(2,2),strides=2,padding='same',name=name+"-max-pool")

    def call(self,input,training=None):
        x1 = self.convo_block(input,training=training)
        x2 = self.pool(x1)
        return (x1,x2)

class Up_Convo_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-convo",filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Up_Convo_Block,self).__init__(name=name,**kwargs)
        self.normalize = normalize
        self.up_convo1 = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_init, name=name+"-convo2D")
        self.norm1 = BatchNormalization(name=name+"-batchNorm")
    
    # Expose training:
    # - BatchNorm -> performs differently when predicting
    def call(self,inputs,training = None, **kwargs):
        x = self.up_convo1(inputs)
        if self.normalize:
            x = self.norm1(x,training=training)
        return x

class Up_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Up_Block,self).__init__(name=name,**kwargs)  
        self.up_convo1 = Up_Convo_Block(name=name+"-up-convo",filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.convo_block1 = Convo_Block(name+"-convo-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.concat = Concatenate(axis=3,name=name+"-concat")

    def call(self, inputs, merger, training=None,**kwargs):
        x = self.up_convo1(inputs,training=training)
        x = self.concat([x,merger])
        return self.convo_block1(x,training=training)

class Attention_Block(tf.keras.layers.Layer):
    def __init__(self,name="Attention-block",filters=64,normalize=False,**kwargs):
        super(Attention_Block,self).__init__(name=name,**kwargs)
        self.normalize = normalize
        self.theta_x = Conv2D(filters=filters, kernel_size=1, strides=(1,1), padding='same',name=name+"-conv2D-1")
        self.norm1 = BatchNormalization(name=name+"-batchNorm-1")
        self.phi_g = Conv2D(filters=filters, kernel_size=1, strides=(1,1), padding='same',name=name+"-conv2D-2")
        self.norm2 = BatchNormalization(name=name+"-batchNorm-2")
        self.add = Add(name=name+"-add")
        self.f = Activation(activation='relu',name=name+"-activ-1")
        self.psi_f = Conv2D(filters=1,kernel_size=1,strides=(1,1),padding='same',name=name+"-conv2D-3")
        self.norm3 = BatchNormalization(name=name+"-batchNorm-3")
        self.activ1 = Activation(activation='sigmoid',name=name+"-activ-2")
        self.activ2 = Activation(activation='sigmoid',name=name+"-activ-3")
        self.att_x = Multiply(name=name+"-multiply")

    # Expose training:
    # - BatchNorm -> performs differently when predicting
    def call(self,x,g,training=None, **kwargs):
        theta_x = self.theta_x(x)
        if self.normalize:
            theta_x = self.norm1(theta_x,training=training)
        phi_g = self.phi_g(g)
        if self.normalize:
            phi_g = self.norm2(phi_g,trainin=training)
        add = self.add([phi_g,theta_x])
        f = self.activ1(add)
        psi_f = self.psi_f(f)
        if self.normalize:
            psi_f = self.norm3(psi_f,training=training)
        rate = self.activ2(psi_f)
        return self.att_x([x,rate])

class Attention_Block_Up(tf.keras.layers.Layer):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Attention_Block_Up,self).__init__(name=name,**kwargs)
        self.up = Up_Convo_Block(name=name+"-up-conv",filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.att = Attention_Block(name=name+"-attention-block",filters=filters,normalize=normalize)
        self.concat = Concatenate(axis=3,name=name+"-concat")
        self.convo = Convo_Block(name=name+"-convo-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    # Expose training
    def call(self, down_layer, input, training=None, **kwargs):
        up = self.up(inputs=down_layer,training=training)
        att = self.att(x=input,g=up,training=training)
        concat = self.concat([up,att])
        return self.convo(inputs=concat,training=training)