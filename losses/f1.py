import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
import keras.backend as K
import tensorflow_addons as tfa
import torch

def f1_score_torch(prediction, targets):
    # best value is at 1, worst at 0
    targets = targets.cpu().numpy().squeeze()
    prediction = prediction.detach().cpu().numpy().squeeze()
    prediction = np.round(prediction)
    f1 = f1_score(targets, prediction, average="micro") # calculate the metrics globally
    return f1
    
def f1_loss_tf(targets, prediction):
    prediction = K.flatten(tf.cast(prediction,tf.float32))
    targets = K.flatten(tf.cast(targets,tf.float32))
    #best value is at 1, worst at 0
    f1_score = tfa.metrics.F1Score(num_classes=1)
    return f1_score