import keras
import tensorflow as tf
import keras.backend as K

def DiceBCELoss(targets, inputs, smooth=1e-6):       
    inputs  =   K.flatten(tf.cast(inputs,tf.float32))
    targets =   K.flatten(tf.cast(targets,tf.float32))
    
    BCE =  K.binary_crossentropy(targets, inputs)
    intersection = K.sum(targets*inputs)    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE