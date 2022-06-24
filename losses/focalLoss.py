import tensorflow as tf
import keras.backend as K

# Paper: https://arxiv.org/pdf/1708.02002.pdf
def FocalLoss(alpha=.25, gamma=2.):    
    
    def loss(targets, inputs):
        inputs  =   K.flatten(tf.cast(inputs,tf.float32))
        targets =   K.flatten(tf.cast(targets,tf.float32))
        
        BCE = K.binary_crossentropy(targets, inputs,from_logits=False)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        
        return focal_loss
        
    return loss