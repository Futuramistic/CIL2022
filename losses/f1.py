import keras.backend as K
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

from .loss_harmonizer import *

def f1_score_torch(thresholded_prediction, targets):
    # TODO: find a way of performing these calculations on the GPU, instead of having to call .cpu()

    # best value is at 1, worst at 0
    targets = torch_collapse_channel_dim(targets, take_argmax=True)
    thresholded_prediction = torch_collapse_channel_dim(thresholded_prediction, take_argmax=True)
    test = torch_expand_channel_dim(targets)

    f1 = f1_score(targets, thresholded_prediction, average="micro")  # calculate the metrics globally
    return f1
    
def f1_score_tf(thresholded_prediction, targets):
    # TODO: find a way of performing these calculations on the GPU, instead of having to call .cpu()

    if not isinstance(thresholded_prediction, np.ndarray):
        prediction = thresholded_prediction.numpy()
    else:
        prediction = thresholded_prediction
    prediction = prediction.squeeze()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    targets = targets.squeeze()
    f1 = f1_score(targets, prediction, average="micro") # calculate the metrics globally
    return f1
