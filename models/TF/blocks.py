import tensorflow as tf
import tensorflow.keras as K

from tensorflow.keras.layers import *


"""
This file contains many Blocks used in various models
"""


class ConvoRelu_Block(tf.keras.layers.Layer):
    def __init__(self,name="ConvoRelu-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,
                 kernel_regularizer=K.regularizers.l2(),kernel_size=3,dilation_rate=1,activation='leaky_relu',**kwargs):
        super(ConvoRelu_Block, self).__init__(name=name,**kwargs)
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.convo = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.kernel_init,
                            name=name+"-conv2D", kernel_regularizer=self.kernel_regularizer,dilation_rate=self.dilation_rate)
        self.norm = BatchNormalization(name=self.name+"-batchNorm", axis=3)
        self.actv = Activation(name=self.name+"-activation",activation=self.activation)
        self.drop = Dropout(rate=self.dropout,name=self.name+"-drop")
    
    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, training=None, **kwargs):
        x = self.convo(inputs)
        if self.normalize:
            x = self.norm(x,training)
        x = self.actv(x)
        x = self.drop(x,training)
        return x
    
    def get_config(self):
        base_config = super(ConvoRelu_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['dilation_rate'] = self.dilation_rate
        base_config['kernel_size'] = self.kernel_size
        base_config['activation'] = self.activation 
        return base_config


class Convo_Block(tf.keras.layers.Layer):
    def __init__(self, name="convo-block", dropout=0.5, filters=64, kernel_init='he_normal',
                 kernel_regularizer=K.regularizers.l2(),
                 normalize=False, kernel_size=3, dilation_rate=1, activation='leaky_relu', **kwargs):
        super(Convo_Block, self).__init__(name=name, **kwargs)
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.convorelu1 = ConvoRelu_Block(name=self.name+"-convoRelu-1", dropout=self.dropout, filters=self.filters,
                                          kernel_init=self.kernel_init,
                                          normalize=self.normalize, kernel_regularizer=self.kernel_regularizer,
                                          kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, activation=self.activation)
        self.convorelu2 = ConvoRelu_Block(name=self.name+"-convoRelu-2", dropout=self.dropout, filters=self.filters,
                                          kernel_init=self.kernel_init,
                                          normalize=self.normalize, kernel_regularizer=self.kernel_regularizer,
                                          kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, activation=self.activation)

    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, **kwargs):
        x = self.convorelu1(inputs, **kwargs)
        return self.convorelu2(x, **kwargs)
    
    def get_config(self):
        base_config = super(Convo_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['dilation_rate'] = self.dilation_rate
        base_config['kernel_size'] = self.kernel_size
        base_config['activation'] = self.activation 
        return base_config

class Down_Block(tf.keras.layers.Layer):
    def __init__(self,name="down-block",dropout=0.5,filters=64,kernel_init='he_normal',
                 kernel_regularizer=K.regularizers.l2(),
                 normalize=False,kernel_size=3,dilation_rate=1,activation='leaky_relu',**kwargs):
        super(Down_Block,self).__init__(name=name,**kwargs)
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.convo_block = Convo_Block(name=self.name+"-convo-block", dropout=self.dropout, filters=self.filters,
                                          kernel_init=self.kernel_init,
                                          normalize=self.normalize, kernel_regularizer=self.kernel_regularizer,
                                          kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, activation=self.activation)
        self.pool = MaxPool2D(pool_size=(2,2),strides=2,padding='same',name=self.name+"-max-pool")
    
    # Expose training
    def call(self,input,**kwargs):
        x1 = self.convo_block(input,**kwargs)
        x2 = self.pool(x1,**kwargs)
        return (x1,x2)
    
    def get_config(self):
        base_config = super(Down_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['dilation_rate'] = self.dilation_rate
        base_config['kernel_size'] = self.kernel_size
        base_config['activation'] = self.activation 
        return base_config


class Down_Block_LearnablePool(tf.keras.layers.Layer):
    """
    A learnable pooling layer.

    Similar to a standard pooling layer but instead of choosing between Max Pool and Average Pool, we let the network
    learn which one is best.
    """
    def __init__(self, name="down-block-learnable-pool", dropout=0.5, filters=64, kernel_init='he_normal',
                 kernel_regularizer=K.regularizers.l2(), normalize=False,
                 kernel_size=3, dilation_rate=1, **kwargs):
        super(Down_Block_LearnablePool, self).__init__(name=name, **kwargs)
        self.convo_block = Convo_Block(name + "-convo-block", dropout=dropout, filters=filters, kernel_init=kernel_init,
                                       normalize=normalize, kernel_regularizer=kernel_regularizer,
                                       kernel_size=kernel_size,dilation_rate=dilation_rate)
        self.Lambda = tf.Variable(initial_value=tf.random.normal(()), trainable=True)
        self.avgpool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same', name=name + "-max-pool")
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name=name + "-max-pool")

    # Expose training
    def call(self, input, **kwargs):
        fac = tf.sigmoid(self.Lambda)  # want to bound lambda in [0, 1]
        x1 = self.convo_block(input, **kwargs)
        x2 = fac * self.avgpool(x1, **kwargs) + (1 - fac) * self.maxpool(x1, **kwargs)
        return (x1, x2)


class Transpose_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-convo",filters=64,dropout=0.5,kernel_init='he_normal',normalize=False,
                 kernel_regularizer=K.regularizers.l2(),activation='leaky_relu',**kwargs):
        super(Transpose_Block,self).__init__(name=name,**kwargs)
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.transpose = Conv2DTranspose(filters=self.filters,kernel_size=(4, 4),strides=(2, 2), padding='same',
                                         kernel_initializer=self.kernel_init,name=self.name+"-convo2DTranspose",
                                         kernel_regularizer=self.kernel_regularizer)
        self.norm = BatchNormalization(name=self.name+"-batchNorm",axis=3)
        self.actv = Activation(activation=self.activation, name=self.name+"-activ")
        self.drop = Dropout(rate=self.dropout,name=self.name+"-drop")
    
    # Expose training:
    # - Dropout -> only performed while training
    # - BatchNorm -> performs differently when predicting
    def call(self, inputs, training=None, **kwargs):
        x = self.transpose(inputs)
        if self.normalize:
            x = self.norm(x,training)
        x = self.actv(x)
        x = self.drop(x,training)
        return x

    def get_config(self):
        base_config = super(Transpose_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['activation'] = self.activation 
        return base_config


class UpSampleConvo_Block(tf.keras.layers.Layer):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',
                 kernel_regularizer=K.regularizers.l2(),normalize=False,activation='leaky_relu',**kwargs):
        super(UpSampleConvo_Block,self).__init__(name=name,**kwargs)
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.up = UpSampling2D(name=self.name+"-upSample2D",size=(2,2))
        self.conv = ConvoRelu_Block(name=self.name+"-convoRelu-block",dropout=self.dropout,filters=self.filters,
                                    kernel_init=self.kernel_init,normalize=self.normalize,kernel_regularizer=self.kernel_regularizer,activation=self.activation)

    # Expose training
    def call(self, inputs,**kwargs):
        up = self.up(inputs,**kwargs)
        return self.conv(up,**kwargs)
    
    def get_config(self):
        base_config = super(UpSampleConvo_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['activation'] = self.activation 
        return base_config


class Up_Block(tf.keras.layers.Layer):
    def __init__(self,name="up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,
                 up_convo = Transpose_Block, kernel_regularizer=K.regularizers.l2(),activation='leaky_relu',**kwargs):
        super(Up_Block,self).__init__(name=name,**kwargs) 
        self.dropout = dropout
        self.filters = filters
        self.normalize = normalize
        self.kernel_init = kernel_init
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation 
        self.up_convo_block = up_convo
        self.up_convo = self.up_convo_block(name=self.name+"-up-convo",filters=self.filters,dropout=self.dropout,
                                 kernel_init=self.kernel_init,normalize=self.normalize,kernel_regularizer=self.kernel_regularizer)
        self.convorelu_block = ConvoRelu_Block(name=self.name+"-convoRelu-block",dropout=self.dropout,
                                               filters=self.filters,kernel_init=self.kernel_init,normalize=self.normalize,
                                               kernel_regularizer=self.kernel_regularizer)
        self.concat = Concatenate(axis=3,name=self.name+"-concat")

    #Expose training
    def call(self, x, merger,**kwargs):
        x = self.up_convo(x,**kwargs)
        # original line was "merger.append(x)"
        # changed due to a bug with the modification of input variables to the "call" function of
        # tf.keras.layer.Layer subclasses causing crashes when saving model checkpoints
        merger = [*merger, x]
        x = self.concat(merger,**kwargs)
        return self.convorelu_block(x,**kwargs)

    def get_config(self):
        base_config = super(Up_Block, self).get_config()
        base_config['dropout'] = self.dropout
        base_config['filters'] = self.filters
        base_config['normalize'] = self.normalize
        base_config['kernel_init'] = self.kernel_init
        base_config['kernel_regularizer'] = self.kernel_regularizer
        base_config['activation'] = self.activation 
        base_config['up_convo'] = self.up_convo_block 
        return base_config

class Attention(tf.keras.layers.Layer):
    def __init__(self,name="attention",filters=64,normalize=False,kernel_init='he_normal',
                 kernel_regularizer=K.regularizers.l2(),**kwargs):
        super(Attention,self).__init__(name=name,**kwargs)
        self.normalize = normalize
        self.theta_x = Conv2D(filters=filters, kernel_size=1,kernel_initializer=kernel_init,
                              strides=(1,1), padding='same',kernel_regularizer=kernel_regularizer,name=name+"-conv2D-1")
        self.norm1 = BatchNormalization(name=name+"-batchNorm-1")
        self.phi_g = Conv2D(filters=filters, kernel_size=1,kernel_initializer=kernel_init,
                            strides=(1,1), padding='same',kernel_regularizer=kernel_regularizer,name=name+"-conv2D-2")
        self.norm2 = BatchNormalization(name=name+"-batchNorm-2")
        self.add = Add(name=name+"-add")
        self.f = Activation(activation='relu',name=name+"-relu")
        self.psi_f = Conv2D(filters=1,kernel_size=1,kernel_initializer=kernel_init,strides=(1,1),
                            padding='same',kernel_regularizer=kernel_regularizer,name=name+"-conv2D-3")
        self.norm3 = BatchNormalization(name=name+"-batchNorm-3")
        self.activ = Activation(activation='sigmoid',name=name+"-sigmoid")
        self.att_x = Multiply(name=name+"-multiply")

    # Expose training:
    # - BatchNorm -> performs differently when predicting
    # NO DROPOUT (!)
    def call(self,x,g, **kwargs):
        theta_x = self.theta_x(x,**kwargs)
        if self.normalize:
            theta_x = self.norm1(theta_x,**kwargs)
        phi_g = self.phi_g(g,**kwargs)
        if self.normalize:
            phi_g = self.norm2(phi_g,**kwargs)
        add = self.add([phi_g,theta_x],**kwargs)
        f = self.f(add,**kwargs)
        psi_f = self.psi_f(f,**kwargs)
        if self.normalize:
            psi_f = self.norm3(psi_f,**kwargs)
        rate = self.activ(psi_f,**kwargs)
        return self.att_x([x,rate],**kwargs)


class Attention_Block_Concat(tf.keras.layers.Layer):
    def __init__(self,name="attention-up-block",dropout=0.5,filters=64,kernel_init='he_normal',normalize=False,
                 up_convo=None,kernel_regularizer=K.regularizers.l2(),**kwargs):
        super(Attention_Block_Concat,self).__init__(name=name,**kwargs)
        self.up = up_convo(name=name+"-up-convo",filters=filters,dropout=dropout,kernel_init=kernel_init,
                           normalize=normalize,kernel_regularizer=kernel_regularizer)
        self.att = Attention(name=name+"-attention",filters=filters,normalize=normalize,kernel_init=kernel_init,
                             kernel_regularizer=kernel_regularizer)
        self.concat = Concatenate(axis=3,name=name+"-concat")
        self.convo = ConvoRelu_Block(name=name+"-convoRelu-block",dropout=dropout,filters=filters,
                                     kernel_init=kernel_init,normalize=normalize,kernel_regularizer=kernel_regularizer)

    # Expose training
    def call(self, x, g, training=None, **kwargs):
        up_g = self.up(inputs=g,**kwargs)
        att_x = self.att(x=x,g=up_g,**kwargs)
        concat = self.concat([up_g,att_x],**kwargs)
        return self.convo(inputs=concat,**kwargs)


class Attention_Block(tf.keras.layers.Layer):
    def __init__(self,name="attention-block", dropout=0.5, filters=64, kernel_init='he_normal', normalize=False,
                 up_convo=None, kernel_regularizer=K.regularizers.l2(), **kwargs):
        super(Attention_Block, self).__init__(name=name, **kwargs)
        self.up = up_convo(name=name+"-up-convo",dropout=dropout, filters=filters, kernel_init=kernel_init,
                           normalize=normalize, kernel_regularizer=kernel_regularizer)
        self.att = Attention(name=name+"-attention", filters=filters, normalize=normalize,
                             kernel_regularizer=kernel_regularizer)

    # Expose training
    def call(self, x, g, training=None, **kwargs):
        up_g = self.up(inputs=g, **kwargs)
        return self.att(x=x, g=up_g, **kwargs)


class Attention_PlusPlus_Block(tf.keras.layers.Layer):
    """
    Attention Block used in the Attention UNet++
    """
    def __init__(self,name="attention-plus-plus", dropout=0.5, filters=64, kernel_init='he_normal', normalize=False,
                 up_convo=None,kernel_regularizer=K.regularizers.l2(), **kwargs):
        super(Attention_PlusPlus_Block, self).__init__(name=name, **kwargs)

        self.att = Attention_Block(name=name+"-att-block", filters=filters, kernel_init=kernel_init,
                                   normalize=normalize,
                                   dropout=dropout, up_convo=up_convo, kernel_regularizer=kernel_regularizer)
        self.up = up_convo(name=name+"-up-convo-block", filters=filters, kernel_init=kernel_init, normalize=normalize,
                           kernel_regularizer=kernel_regularizer)
        self.concat = Concatenate(name=name+"-concat", axis=3)
        self.maxpool = MaxPool2D(name=name+"-max")
        self.convo = Convo_Block(name=name+"-convo-block", dropout=dropout, filters=filters, kernel_init=kernel_init,
                                 normalize=normalize, kernel_regularizer=kernel_regularizer)

    def call(self,x,g,down=None,to_concat=[], training=None, **kwargs):
        att_x = self.att(x=x, g=g, **kwargs)
        # original line was "to_concat.append(att_x)"
        # changed due to a bug with the modification of input variables to the "call" function of
        # tf.keras.layer.Layer subclasses causing crashes when saving model checkpoints
        # after another list has been created using the * operator, we can safely use "append" again
        to_concat = [*to_concat, att_x]
        up_g = self.up(g,**kwargs)
        to_concat.append(up_g)
        if down is not None:
            down = self.maxpool(down, **kwargs)
            to_concat.append(down)
        conv = self.concat(to_concat, **kwargs)
        conv = self.convo(conv, **kwargs)
        return att_x, conv
