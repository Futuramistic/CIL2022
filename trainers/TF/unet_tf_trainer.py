import tensorflow as tf
import tensorflow.keras as K

from losses import FocalTverskyLoss
from trainers.trainer_tf import TFTrainer
from utils import *


class UNetTFTrainer(TFTrainer):
    """
    Trainer for the UnetTF model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, pre_process="vgg", use_channelwise_norm=False,
                 blobs_removal_threshold=0, hyper_seg_threshold=False, use_sample_weighting=False, use_adaboost=False,
                 deep_adaboost=False, f1_threshold_to_log_checkpoint=DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT):
        """
        Set omitted parameters to model-specific defaults, then call superclass __init__ function
        @Warning: some arguments depend on others not being None, so respect this order!

        Args:
            Refer to the TFTrainer superclass for more details on the arguments
        """

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        # Large batch size used online: 32
        # Possible overkill
        if batch_size is None:
            batch_size = 2

        train_set_size, test_set_size, unlabeled_test_set_size = dataloader.get_dataset_sizes(split=split)
        steps_per_training_epoch = train_set_size // batch_size

        if num_epochs is None:
            num_epochs = math.ceil(100000 / steps_per_training_epoch)

        if optimizer_or_lr is None:
            # CAREFUL! Smaller learning rate recommended in comparision to other models !!!
            # Even 1e-5 was recommended, but might take ages
            optimizer_or_lr = UNetTFTrainer.get_default_optimizer_with_lr(lr=1e-3)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = UNetTFTrainer.get_default_optimizer_with_lr(lr=optimizer_or_lr)

        # According to the online github repo
        if loss_function is None:
            def get_focal_tversky_loss():
                return lambda x,y: FocalTverskyLoss(alpha=0.3,beta=0.7,gamma=0.75)(x, y)
            loss_function = get_focal_tversky_loss
            # loss_function = K.losses.BinaryCrossentropy(from_logits=False,
            #                                            reduction=K.losses.Reduction.SUM_OVER_BATCH_SIZE)

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(batch_size)

        # convert model input to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in {0, 1} & remove A channel

        # note: no batch dim
        if pre_process == "vgg":
            preprocessing =\
                lambda x, is_gt: x[:, :, :3] if not is_gt \
                else x[:, :, :1] // 255
        # Expect some preprocessing on the model side (VGG architecture preprocessing or other)
        else:
            preprocessing = None
            if use_channelwise_norm and dataloader.dataset in DATASET_STATS:
                def channelwise_preprocessing(x, is_gt):
                    if is_gt:
                        return x[:, :, :1] // 255
                    stats = DATASET_STATS[dataloader.dataset]
                    x = tf.cast(x[:, :, :3], dtype=tf.float32)
                    x = tf.concat([(x[:, :, 0] - stats['pixel_mean_0']) / stats['pixel_std_0'],
                                (x[:, :, 1] - stats['pixel_mean_1']) / stats['pixel_std_1'],
                                (x[:, :, 2] - stats['pixel_mean_2']) / stats['pixel_std_2']], axis=-1)
                    return x
                preprocessing = channelwise_preprocessing
            else:
                # convert samples to float32 \in [0, 1] & remove A channel;
                # convert ground truth to int \in {0, 1} & remove A channel
                preprocessing =\
                    lambda x, is_gt: (tf.cast(x[:, :, :3], dtype=tf.float32) / 255.0) if not is_gt \
                    else (x[:, :, :1] // 255)
                    
        super().__init__(dataloader, model, preprocessing, steps_per_training_epoch, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold, use_channelwise_norm, blobs_removal_threshold, False,  # hyper_seg_threshold
                         use_sample_weighting, use_adaboost, deep_adaboost, f1_threshold_to_log_checkpoint)

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        """
        return {**(super()._get_hyperparams()),
                **({param: getattr(self.model, param)
                   for param in ['dropout', 'kernel_init', 'normalize', 'kernel_regularizer', 'up_transpose']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr):
        """
        Return the default optimizer for this network.
        Args:
            lr (float): Learning rate of the optimizer
            model: Model whose parameters we want to train

        @Note: No mention on learning rate decay; can be reintroduced
        """
        # lr_schedule = K.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_rate=0.1,
        #                                                      decay_steps=30000, staircase=True)
        return K.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
