import keras
import tensorflow as tf
import keras.backend as K

# Two implementations can be found online:
# -> BCE + dice_loss
# -> 0.5*BCE - dice_coeff

# BCE + dice_loss
def DiceBCELoss1(targets, inputs, smooth=1e-6):       
    inputs  =   K.flatten(tf.cast(inputs,tf.float32))
    targets =   K.flatten(tf.cast(targets,tf.float32))
    
    BCE =  K.binary_crossentropy(targets, inputs, from_logits=True)
    intersection = K.sum(targets*inputs)    
    dice_coeff = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    dice_loss = 1 - dice_coeff
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

# 0.5*BCE - dice_coeff
def DiceBCELoss2(targets, inputs, smooth=1e-6):       
    inputs  =   K.flatten(tf.cast(inputs,tf.float32))
    targets =   K.flatten(tf.cast(targets,tf.float32))
    
    BCE =  K.binary_crossentropy(targets, inputs, from_logits=True)
    intersection = K.sum(targets*inputs)    
    dice_coeff = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = 0.5*BCE + dice_coeff
    return Dice_BCE