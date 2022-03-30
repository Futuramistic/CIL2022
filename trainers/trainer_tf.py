import abc
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import mlflow_logger
from .trainer import Trainer
from utils import *


class TFTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing, steps_per_training_epoch,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None):
        """
        Abstract class for TensorFlow-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, evaluation_interval, num_samples_to_visualize, checkpoint_interval)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.steps_per_training_epoch = steps_per_training_epoch

    class Callback(K.callbacks.Callback):
        def __init__(self, trainer, mlflow_run):
            super().__init__()
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_begin(self, logs=None):
            pass

        def on_train_end(self, logs=None):
            pass

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs=None):
            mlflow_logger.log_metrics(logs)

        def on_train_batch_begin(self, batch, logs=None):
            pass

        def on_train_batch_end(self, batch, logs=None):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                if self.do_visualize:
                    mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
            self.iteration_idx += 1

    def create_visualizations(self):
        images = []
        num_samples = self.num_samples_to_visualize

        for batch_xs, batch_ys in self.test_loader.shuffle(10 * num_samples).batch(num_samples):
            batch_xs = tf.squeeze(batch_xs, axis=1)
            batch_ys = tf.squeeze(batch_ys, axis=1).numpy()
            output = self.model.predict(batch_xs)
            preds = tf.argmax(output, axis=-1).numpy()  # taking the argmax removes the last dim
            preds = np.expand_dims(preds, axis=1)  # so add it back, in CHW format
            batch_ys = np.moveaxis(batch_ys, -1, 1)  # TODO only do this if we know the network uses HWC format
            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            if batch_ys is None:
                batch_ys = np.zeros_like(preds)
            merged = (2 * preds + batch_ys)
            green_channel = merged == 3  # true positives
            red_channel = merged == 2  # false positive
            blue_channel = merged == 1  # false negative
            rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=1)
            for batch_sample_idx in range(batch_xs.shape[0]):
                images.append(rgb[batch_sample_idx])
            break  # Only operate on one batch of 'self.trainer.num_samples_to_visualize' samples
        return images

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer_or_lr)

    def _fit_model(self, mlflow_run):
        self._compile_model()
        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing)
        self.test_loader = self.dataloader.get_testing_dataloader(split=self.split, batch_size=1,
                                                                  preprocessing=self.preprocessing)
        callbacks = [TFTrainer.Callback(self, mlflow_run)]
        if self.do_checkpoint:
            checkpoint_path = "{dir}".format(dir=CHECKPOINTS_DIR) + "cp-{epoch:04d}.ckpt"
            checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                  save_freq=self.checkpoint_interval * self.steps_per_training_epoch)
            callbacks.append(checkpoint_callback)
        last_test_loss = self.model.fit(self.train_loader, epochs=self.num_epochs,
                                        steps_per_epoch=self.steps_per_training_epoch, callbacks=callbacks)
        return last_test_loss
