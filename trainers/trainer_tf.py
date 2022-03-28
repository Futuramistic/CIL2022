import abc
import itertools
import mlflow
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import tensorflow.keras as K
import time

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
            self.testing_dl = self.trainer.dataloader.get_testing_dataloader(split=self.trainer.split, batch_size=1,
                                                                             preprocessing=self.trainer.preprocessing)
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_begin(self, logs=None):
            pass

        def on_train_end(self, logs=None):
            pass

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs=None):
            pass

        def on_train_batch_begin(self, batch, logs=None):
            pass

        def on_train_batch_end(self, batch, logs=None):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                # remember: dataloader may be infinite
                if self.do_visualize:
                    mlflow_logger.log_visualizations(self)
            self.iteration_idx += 1

        def create_visualizations(self, temp_dir):
            def segmentation_to_image(x):
                x = (x * 255).astype(int)
                if len(x.shape) < 3:
                    x = np.expand_dims(x, axis=-1)
                return x

            n = self.trainer.num_samples_to_visualize
            if is_perfect_square(n):
                nb_cols = math.sqrt(n)
            else:
                nb_cols = math.sqrt(next_perfect_square(n))
            nb_cols = int(nb_cols)  # Need it to be an integer
            images = []

            # TODO batch_xs does not actually contain a batch, only a single image
            # It is just iterating over the selected images, which is not optimal performance-wise
            for batch_xs, batch_ys in self.testing_dl.take(self.trainer.num_samples_to_visualize):
                output = self.model.predict(batch_xs)
                preds = tf.argmax(output, axis=-1)
                if batch_ys is None:
                    batch_ys = tf.zeros_like(preds)
                batch_ys = tf.squeeze(batch_ys, axis=-1)
                preds = tf.cast(preds, dtype=tf.uint8)
                merged = (2 * preds + batch_ys).numpy()
                green_channel = merged == 3  # true positives
                red_channel = merged == 2  # false positive
                blue_channel = merged == 1  # false negative
                rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=0)
                images.append(rgb)

            nb_rows = math.ceil(float(n) / float(nb_cols))  # Number of rows in final image
            # Append enough black images to complete the last non-empty row
            while len(images) < nb_cols * nb_rows:
                images.append(tf.zeros_like(images[0]))
            arr = []  # Store images concatenated in the last dimension here
            for i in range(nb_rows):
                row = np.concatenate(images[(i * nb_cols):(i + 1) * nb_cols], axis=-1)
                arr.append(row)
            # Concatenate in the second-to-last dimension to get the final big image
            final = np.concatenate(arr, axis=-2)
            K.preprocessing.image.save_img(os.path.join(temp_dir, f'rgb.png'),
                                           segmentation_to_image(final), data_format="channels_first")

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer_or_lr)

    def _fit_model(self, mlflow_run):
        dataset = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                          preprocessing=self.preprocessing)
        last_test_loss = self.model.fit(dataset, epochs=self.num_epochs, steps_per_epoch=self.steps_per_training_epoch,
                       callbacks=[TFTrainer.Callback(self, mlflow_run)])
        return last_test_loss

    def _train(self):
        if self.mlflow_experiment_name is not None:
            self._init_mlflow()
            self._compile_model()
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_experiment_name) as run:
                mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                self._fit_model(mlflow_run=run)
                mlflow_logger.log_codebase()
        else:
            self._compile_model()
            return self._fit_model(mlflow_run=None)
