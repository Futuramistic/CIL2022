import tensorflow as tf
from keras.layers import *

class ConvoRelu_Block(tf.keras.layers.Layer):
    def __init__(self,name="ConvoRelu-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(ConvoRelu_Block, self).__init__(name=name,**kwargs)
        self.normalize = normalize  
        self.convo = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=kernel_init,name=name+"-conv2D")
        self.norm = BatchNormalization(name=name+"-batchNorm")
        self.actv = Activation(activation='relu',name=name+"-activ")
        self.drop = Dropout(rate=dropout,name=name+"-drop")
    
    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, training=None, **kwargs):
        x = self.convo(inputs)
        if self.normalize:
            x = self.norm(x,training=training)
        x = self.actv(x)
        if training:
            x = self.drop(x)
        return x

class Convo_Block(tf.keras.layers.Layer):
    def __init__(self,name="convo-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Convo_Block, self).__init__(name=name,**kwargs)     
        self.convorelu1 = ConvoRelu_Block(name=name+"-convoRelu-1",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.convorelu2 = ConvoRelu_Block(name=name+"-convoRelu-2",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, training=None, **kwargs):
        x = self.convorelu1(inputs,training,**kwargs)
        return self.convorelu2(x,training,**kwargs)

class Down_Block(tf.keras.layers.Layer):
    def __init__(self,name="down-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(Down_Block,self).__init__(name=name,**kwargs)
        self.convo_block = Convo_Block(name+"-convo-block",dropout,filters,kernel_init,normalize)
        self.pool = MaxPool2D(pool_size=(2,2),strides=2,padding='same',name=name+"-max-pool")
    
    # Expose training
    def call(self,input,training=None):
        x1 = self.convo_block(input,training=training)
        x2 = self.pool(x1)
        return (x1,x2)

class Transpose_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-convo",filters=64,dropout=0.5,kernel_init='he_normal',normalize=False,**kwargs):
        super(Transpose_Block,self).__init__(name=name,**kwargs)
        self.normalize = normalize
        self.transpose = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_init, name=name+"-convo2DTranspose")
        self.norm = BatchNormalization(name=name+"-batchNorm")
        self.actv = Activation(activation='relu', name=name+"-activ")
        self.drop = Dropout(rate=dropout,name=name+"-drop")
    
    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self,inputs,training = None, **kwargs):
        x = self.transpose(inputs)
        if self.normalize:
            x = self.norm(x,training=training)
        x = self.actv(x)
        if training:
            x = self.drop(x)
        return x

class UpSampleConvo_Block(tf.keras.layers.Layer):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,**kwargs):
        super(UpSampleConvo_Block,self).__init__(name=name,**kwargs)
        self.up = UpSampling2D(name=name+"-upSample2D",size=(2,2))
        self.conv = ConvoRelu_Block(name=name+"-convoRelu-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    # Expose training
    def call(self, inputs, training=None, **kwargs):
        up = self.up(inputs,training=training, **kwargs)
        return self.conv(up,training=training, **kwargs)

class Up_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False, up_convo = None,**kwargs):
        super(Up_Block,self).__init__(name=name,**kwargs)  
        self.up_convo = up_convo(name=name+"-up-convo",filters=filters,dropout=dropout,kernel_init=kernel_init,normalize=normalize)
        self.convorelu_block = ConvoRelu_Block(name+"-convoRelu-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.concat = Concatenate(axis=3,name=name+"-concat")

    #Expose training
    def call(self, inputs, merger, training=None,**kwargs):
        x = self.up_convo(inputs,training=training)
        x = self.concat([x,merger])
        return self.convorelu_block(x,training=training)

class Attention(tf.keras.layers.Layer):
    def __init__(self,name="attention",filters=64,normalize=False,kernel_init='he_normal',**kwargs):
        super(Attention,self).__init__(name=name,**kwargs)
        self.normalize = normalize
        self.theta_x = Conv2D(filters=filters, kernel_size=1,kernel_initializer=kernel_init,strides=(1,1), padding='same',name=name+"-conv2D-1")
        self.norm1 = BatchNormalization(name=name+"-batchNorm-1")
        self.phi_g = Conv2D(filters=filters, kernel_size=1,kernel_initializer=kernel_init,strides=(1,1), padding='same',name=name+"-conv2D-2")
        self.norm2 = BatchNormalization(name=name+"-batchNorm-2")
        self.add = Add(name=name+"-add")
        self.f = Activation(activation='relu',name=name+"-activ-1")
        self.psi_f = Conv2D(filters=1,kernel_size=1,kernel_initializer=kernel_init,strides=(1,1),padding='same',name=name+"-conv2D-3")
        self.norm3 = BatchNormalization(name=name+"-batchNorm-3")
        self.activ1 = Activation(activation='sigmoid',name=name+"-activ-2")
        self.activ2 = Activation(activation='sigmoid',name=name+"-activ-3")
        self.att_x = Multiply(name=name+"-multiply")

    # Expose training:
    # - BatchNorm -> performs differently when predicting
    # NO DROPOUT (!)
    def call(self,x,g,training=None, **kwargs):
        theta_x = self.theta_x(x)
        if self.normalize:
            theta_x = self.norm1(theta_x,training=training)
        phi_g = self.phi_g(g)
        if self.normalize:
            phi_g = self.norm2(phi_g,training=training)
        add = self.add([phi_g,theta_x])
        f = self.activ1(add)
        psi_f = self.psi_f(f)
        if self.normalize:
            psi_f = self.norm3(psi_f,training=training)
        rate = self.activ2(psi_f)
        return self.att_x([x,rate])

class Attention_Block_Concat(tf.keras.layers.Layer):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,up_convo=None,**kwargs):
        super(Attention_Block_Concat,self).__init__(name=name,**kwargs)
        self.up = up_convo(name=name+"-up-convo",filters=filters,dropout=dropout,kernel_init=kernel_init,normalize=normalize)
        self.att = Attention(name=name+"-attention",filters=filters,normalize=normalize,kernel_init=kernel_init)
        self.concat = Concatenate(axis=3,name=name+"-concat")
        self.convo = ConvoRelu_Block(name=name+"-convoRelu-block",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)

    # Expose training
    def call(self, x, g, training=None, **kwargs):
        up_g = self.up(inputs=g,training=training)
        att_x = self.att(x=x,g=up_g,training=training)
        concat = self.concat([up_g,att_x])
        return self.convo(inputs=concat,training=training)

class Attention_Block(tf.keras.layers.Layer):
    def __init__(self,name="attention-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,up_convo=None,**kwargs):
        super(Attention_Block,self).__init__(name=name,**kwargs)
        self.up = up_convo(name=name+"-up-convo",dropout=dropout,filters=filters,kernel_init=kernel_init,normalize=normalize)
        self.att = Attention(name=name+"-attention",filters=filters,normalize=normalize)

    # Expose training
    def call(self, x, g, training=None, **kwargs):
        up_g = self.up(inputs=g,training=training)
        return self.att(x=x,g=up_g,training=training)