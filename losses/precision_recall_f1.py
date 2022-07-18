from typing import Iterable
from .loss_harmonizer import *
import tensorflow.keras as K
import tensorflow_addons as tfa
import tensorflow as tf

"""
If we take the F1 score for both classes (corresponding to average='micro' for sklearn.metrics.f1_score),
the model can easily get a high F1 score, as most pixels correspond to the background
check which classes are used for evaluation
"""
DEFAULT_F1_CLASSES = (1,)


def prediction_stats_torch(thresholded_prediction, targets, classes, dtype=torch.float32):
    """
    Compute statics of the predictions i.e. the true positives/negatives and false positives/negatives
    Torch version.
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
        dtype: Type of the tensors
    """
    targets = collapse_channel_dim_torch(targets, take_argmax=True).long()
    thresholded_prediction = collapse_channel_dim_torch(thresholded_prediction, take_argmax=True).long()

    tp, fp, tn, fn = [torch.zeros(1, dtype=dtype, device=targets.device) for _ in range(4)]

    if not isinstance(classes, Iterable):
        classes = [classes]

    for class_idx in classes:
        thr_pos = thresholded_prediction == class_idx
        thr_neg = thresholded_prediction != class_idx
        trg_pos = targets == class_idx
        trg_neg = targets != class_idx

        tp += torch.logical_and(thr_pos, trg_pos).sum()
        fp += torch.logical_and(thr_pos, trg_neg).sum()
        tn += torch.logical_and(thr_neg, trg_neg).sum()
        fn += torch.logical_and(thr_neg, trg_pos).sum()

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def precision_torch(thresholded_prediction, targets, classes, pred_stats=None):
    """
    Compute prediction precision
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
        pred_stats: Type of the tensors
    """
    if pred_stats is None:
        pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == torch.zeros_like(pred_stats['tp']):  # prevent NaNs
        return torch.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fp'])


def recall_torch(thresholded_prediction, targets, classes, pred_stats=None):
    """
    Compute prediction recall
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
        pred_stats: Type of the tensors
    """
    if pred_stats is None:
        pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == torch.zeros_like(pred_stats['tp']):  # prevent NaNs
        return torch.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fn'])


def precision_recall_f1_score_torch(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    """
    Compute precision, recall and f1 score
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
    """
    # best value is at 1, worst at 0
    pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    precision = precision_torch(thresholded_prediction, targets, classes, pred_stats)
    recall = recall_torch(thresholded_prediction, targets, classes, pred_stats)
    if precision == torch.zeros_like(precision):  # prevent NaNs
        return precision, recall, torch.zeros_like(precision)
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def f1_score_torch(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    """
    Compute the f1 score
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
    """
    _, _, f1_score = precision_recall_f1_score_torch(thresholded_prediction, targets, classes)
    return f1_score


def prediction_stats_tf(thresholded_prediction, targets, classes, dtype=tf.dtypes.float32):
    """
    Compute statics of the predictions i.e. the true positives/negatives and false positives/negatives
    Tensorflow version.
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
        dtype: Type of the tensors
    """
    targets = collapse_channel_dim_tf(targets, take_argmax=True)
    thresholded_prediction = tf.cast(collapse_channel_dim_tf(thresholded_prediction, take_argmax=True),
                                     dtype=targets.dtype)

    tp, fp, tn, fn = [tf.zeros(1, dtype=dtype) for _ in range(4)]

    if not isinstance(classes, Iterable):
        classes = [classes]

    for class_idx in classes:
        class_idx_tensor = tf.cast(tf.convert_to_tensor(class_idx), dtype=targets.dtype)
        thr_pos = tf.math.equal(thresholded_prediction, class_idx_tensor)
        thr_neg = tf.math.not_equal(thresholded_prediction, class_idx_tensor)
        trg_pos = tf.math.equal(targets, class_idx_tensor)
        trg_neg = tf.math.not_equal(targets, class_idx_tensor)

        tp += tf.math.count_nonzero(tf.math.logical_and(thr_pos, trg_pos), dtype=tp.dtype)
        fp += tf.math.count_nonzero(tf.math.logical_and(thr_pos, trg_neg), dtype=fp.dtype)
        tn += tf.math.count_nonzero(tf.math.logical_and(thr_neg, trg_neg), dtype=tn.dtype)
        fn += tf.math.count_nonzero(tf.math.logical_and(thr_neg, trg_pos), dtype=fn.dtype)

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def precision_tf(thresholded_prediction, targets, target_class):
    """
    Compute prediction precision
    Args:
        thresholded_prediction (TF Tensor): binary prediction
        targets (TF Tensor): The target tensor
        target_class (int): Class to predict for
    """
    precision_metric = K.metrics.Precision(dtype=tf.float32)
    precision_metric.update_state(targets,tf.equal(thresholded_prediction,target_class))
    return precision_metric.result().numpy()


def recall_tf(thresholded_prediction, targets, target_class):
    """
    Compute prediction recall
    Args:
        thresholded_prediction (TF Tensor): binary prediction
        targets (TF Tensor): The target tensor
        target_class (int): Class to predict for
    """
    recall_metric = K.metrics.Recall(dtype=tf.float32)
    recall_metric.update_state(targets,tf.equal(thresholded_prediction,target_class))
    return recall_metric.result().numpy()

def f1_micro_tf(thresholded_prediction,targets,target_class):
    """
    Compute prediction F1 score (micro)
    Args:
        thresholded_prediction (TF Tensor): binary prediction
        targets (TF Tensor): The target tensor
        target_class (int): Class to predict for
    """
    f1_micro_metric = tfa.metrics.F1Score(num_classes=2, average='micro',dtype=tf.float32)
    f1_micro_metric.update_state(targets,tf.cast(tf.equal(thresholded_prediction,target_class),tf.float32))
    return f1_micro_metric.result().numpy()

def tf_count(t, val):
    """
    Count number of occurrences of a class
    Args:
        t (TF Tensor): Tensor to calculate for
        val (int32): value to count to occurrences of
    """
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def precision_recall_f1_score_tf(thresholded_prediction, targets):
    """
    Compute precision, recall and f1 score
    Args:
        thresholded_prediction (TF Tensor): binary prediction
        targets (TF Tensor): The target tensor
    """
    thresholded_prediction = tf.squeeze(thresholded_prediction)
    targets = tf.squeeze(targets)

    precision_road = precision_tf(thresholded_prediction,targets,1)
    precision_bkgd = precision_tf(thresholded_prediction,targets,0)

    recall_road = recall_tf(thresholded_prediction,targets,1)
    recall_bkgd = recall_tf(thresholded_prediction,targets,0)

    f1_micro_road = f1_micro_tf(thresholded_prediction,targets,1)
    f1_micro_bkgd = f1_micro_tf(thresholded_prediction,targets,0)
    
    ones_weight =   tf_count(targets,1).numpy().item()/tf.size(targets,out_type=tf.int32).numpy().item()
    zeros_weight =  tf_count(targets,0).numpy().item()/tf.size(targets,out_type=tf.int32).numpy().item()

    f1_macro = (f1_micro_road+f1_micro_bkgd)/2.0
    f1_weighted = ones_weight * f1_micro_road + zeros_weight * f1_micro_bkgd
    
    return (precision_road, recall_road, f1_micro_road, 
            precision_bkgd, recall_bkgd, f1_micro_bkgd, 
            f1_macro, f1_weighted)

def f1_score_tf(thresholded_prediction, targets):
    """
    Compute the f1 score
    Args:
        thresholded_prediction (Torch Tensor): binary prediction
        targets (Torch Tensor): The target tensor
        classes (list): List of classes for which we want to compute the statistics
    """
    _, _, _, _, _, _, _, f1_weighted = precision_recall_f1_score_tf(thresholded_prediction, targets)
    return f1_weighted