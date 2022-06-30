import math
import os

import mlflow
import tensorflow as tf
import tensorflow.keras as K

from .trainer_tf import TFTrainer
from utils import *


class GLDenseUNetTrainer(TFTrainer):
    """
    Trainer for the GL-Dense-U-Net model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        if batch_size is None:
            batch_size = 8

        train_set_size, test_set_size, unlabeled_test_set_size = dataloader.get_dataset_sizes(split=split)
        steps_per_training_epoch = train_set_size // batch_size

        if num_epochs is None:
            num_epochs = math.ceil(100000 / steps_per_training_epoch)

        if optimizer_or_lr is None:
            optimizer_or_lr = GLDenseUNetTrainer.get_default_optimizer_with_lr(1e-4, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = GLDenseUNetTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if loss_function is None:
            cat_xent = K.losses.SparseCategoricalCrossentropy(from_logits=False)
            loss_function = lambda targets, inputs, cat_xent=cat_xent: cat_xent(tf.squeeze(targets, axis=-1), inputs)
            self.loss_function_name = 'SparseCategoricalCrossentropy'

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs, num_samples_to_visualize)

        # convert samples to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in {0, 1} & remove A channel
        preprocessing =\
            lambda x, is_gt: (tf.cast(x[:, :, :3], dtype=tf.float32) / 255.0) if not is_gt \
            else (x[:, :, :1] // 255)

        super().__init__(dataloader, model, preprocessing, steps_per_training_epoch, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold)

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self.model, param)
                   for param in ['growth_rate', 'layers_per_block', 'conv2d_activation', 'num_classes',
                                 'input_resize_dim', 'l2_regularization_param']
                   if hasattr(self.model, param)})}

    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        # uses learning rate decay; see
        # https://github.com/cugxyy/GL-Dense-U-Net/blob/ce104189692dd8e1a22ddcabc9f2f685a8345806/Model/multi_gpu_train.py
        lr_schedule = K.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_rate=0.1,
                                                              decay_steps=30000, staircase=True)
        return K.optimizers.Adam(lr_schedule)
