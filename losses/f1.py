import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
import keras.backend as K
import tensorflow_addons as tfa

def f1_score_torch(prediction, targets):
    # best value is at 1, worst at 0
    targets = targets.cpu().numpy().squeeze()
    prediction = prediction.detach().cpu().numpy().squeeze()
    prediction = np.round(prediction)
    f1 = f1_score(targets, prediction, average="micro") # calculate the metrics globally
    return f1
    
def f1_score_tf(prediction, targets):
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.numpy()
    prediction = prediction.squeeze()
    prediction = np.round(prediction)
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    targets = targets.squeeze()
    f1 = f1_score(targets, prediction, average="micro") # calculate the metrics globally
    return f1