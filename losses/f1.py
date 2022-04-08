from sklearn.metrics import f1_score
import tensorflow as tf
import keras.backend as K
import tensorflow_addons as tfa

def f1_score_torch(targets, prediction):
    # best value is at 1, worst at 0
    print(targets)
    print("Prediction:")
    print(prediction)
    targets = targets.squeeze()
    targets = targets[1,:,:].squeeze()
    prediction = prediction.squeeze()
    print(targets.shape, prediction.shape)
    f1 = f1_score(targets, prediction)
    return f1
    
def f1_loss_tf(targets, prediction):
    prediction = K.flatten(tf.cast(prediction,tf.float32))
    targets = K.flatten(tf.cast(targets,tf.float32))
    #best value is at 1, worst at 0
    f1_score = tfa.metrics.F1Score(num_classes=1)
    return f1_score