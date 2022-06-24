import keras
import tensorflow as tf
import keras.backend as K

#   Paper: https://arxiv.org/pdf/1706.05721.pdf
def FocalTverskyLoss( alpha=.5, beta=.5, gamma=1, smooth=1e-6):
        
        def loss(targets, inputs):
                #flatten label and prediction tensors
                inputs  =   K.flatten(tf.cast(inputs,tf.float32))
                targets =   K.flatten(tf.cast(targets,tf.float32))
                
                #True Positives, False Positives & False Negatives
                TP = K.sum((inputs * targets))
                FP = K.sum(((1-targets) * inputs))
                FN = K.sum((targets * (1-inputs)))
                
                Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
                FocalTversky = K.pow((1 - Tversky), gamma)
                
                return FocalTversky
                
        return loss