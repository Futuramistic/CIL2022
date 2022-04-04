import tensorflow as tf
from keras.layers import *
import tensorflow.keras as K
from .blocks import *

def UNetTF(input_shape,name="UNetTF",dropout=0.5,kernel_init='he_normal',normalize=False, up_transpose=True):

    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        convo1 = Conv2D(filters=nb_filters[0], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(inputs)
        convo2 = Conv2D(filters=nb_filters[0], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo1)
        
        pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(convo2)
        pool1 = Dropout(dropout)(pool1)

        convo3 = Conv2D(filters=nb_filters[1], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(pool1)
        convo4 = Conv2D(filters=nb_filters[1], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo3)
        pool2 = MaxPool2D(pool_size=(2, 2), strides=2)(convo4)
        pool2 = Dropout(dropout)(pool2)

        convo5 = Conv2D(filters=nb_filters[2], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(pool2)
        convo6 = Conv2D(filters=nb_filters[2], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo5)
        pool3 = MaxPool2D(pool_size=(2, 2), strides=2)(convo6)
        pool3 = Dropout(dropout)(pool3)

        convo7 = Conv2D(filters=nb_filters[3], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(pool3)
        convo8 = Conv2D(filters=nb_filters[3], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo7)
        pool4 = MaxPool2D(pool_size=(2, 2), strides=2)(convo8)
        pool4 = Dropout(dropout)(pool4)

        # BOTTOM
        convo9 = Conv2D(filters=nb_filters[4], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(pool4)
        convo10 = Conv2D(filters=nb_filters[4], kernel_size=3, activation='relu', padding="same",
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo9)

        # GOING UP
        upConvo1 = Conv2DTranspose(filters=nb_filters[4], kernel_size=(3, 3), strides=(2, 2), padding="same",
                                activation='relu', kernel_initializer=kernel_init)(convo10)
        merge1 = Concatenate()([convo8, upConvo1])
        merge1 = Dropout(dropout)(merge1)
        convo11 = Conv2D(filters=nb_filters[3], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(merge1)
        convo12 = Conv2D(filters=nb_filters[3], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo11)

        upConvo2 = Conv2DTranspose(filters=nb_filters[3], kernel_size=(3, 3), strides=(2, 2), padding="same",
                                activation='relu', kernel_initializer=kernel_init)(convo12)
        merge2 = Concatenate()([convo6, upConvo2])
        merge2 = Dropout(dropout)(merge2)
        convo13 = Conv2D(filters=nb_filters[2], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init)(merge2)
        convo14 = Conv2D(filters=nb_filters[2], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init)(convo13)

        upConvo3 = Conv2DTranspose(filters=nb_filters[2], kernel_size=(3, 3), strides=(2, 2), padding="same",
                                activation='relu', kernel_initializer=kernel_init)(convo14)
        merge3 = Concatenate()([convo4, upConvo3])
        merge3 = Dropout(dropout)(merge3)
        convo15 = Conv2D(filters=nb_filters[1], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2())(merge3)
        convo16 = Conv2D(filters=nb_filters[1], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo15)

        upConvo4 = Conv2DTranspose(filters=nb_filters[1], kernel_size=(3, 3), strides=(2, 2), padding="same",
                                activation='relu', kernel_initializer=kernel_init)(convo16)
        merge4 = Concatenate()([convo2, upConvo4])
        merge4 = Dropout(dropout)(merge4)
        convo17 = Conv2D(filters=nb_filters[0], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(merge4)
        convo18 = Conv2D(filters=nb_filters[0], kernel_size=3, padding="same", activation='relu',
                        kernel_initializer=kernel_init, kernel_regularizer=K.regularizers.l2(1e-5))(convo17)

        output = Conv2D(filters=1, kernel_size=(1, 1), padding="same",
                        kernel_initializer=kernel_init, activation='sigmoid')(convo18)
        return output

    
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model