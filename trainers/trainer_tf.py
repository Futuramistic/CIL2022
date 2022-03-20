import abc
import mlflow
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import tensorflow.keras as K

from .trainer import Trainer
from utils import *


class TFTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing, steps_per_training_epoch,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, loss_function=None, evaluation_interval=None):
        """
        Abstract class for TensorFlow-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, evaluation_interval)
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
                testing_dl = self.trainer.dataloader.get_testing_dataloader(split=self.trainer.split, batch_size=1,
                                                                            preprocessing=self.trainer.preprocessing)
                _, test_size, _ = self.trainer.dataloader.get_dataset_sizes(split=self.trainer.split)
                # store segmentations in temp_dir, then upload temp_dir to Mlflow server
                temp_dir = tempfile.mkdtemp()

                def segmentation_to_image(x):
                    x = (x * 255).astype(int)
                    if len(x.shape) < 3:
                        x = np.expand_dims(x, axis=-1)
                    return x

                def loop():
                    sample_idx = 0
                    for batch_xs, batch_ys in testing_dl:
                        logits = self.model.predict(batch_xs)
                        preds = np.argmax(logits, axis=-1)
                        for batch_sample_idx in range(batch_xs.shape[0]):
                            # TODO: merge prediction and ground truth into one image using colors, to save space/traffic
                            K.preprocessing.image.save_img(os.path.join(temp_dir, f'{sample_idx}_pred.png'),
                                                           segmentation_to_image(preds[batch_sample_idx]))
                            if batch_ys is not None:
                                K.preprocessing.image.save_img(os.path.join(temp_dir, f'{sample_idx}_gt.png'),
                                                               segmentation_to_image(batch_ys[batch_sample_idx].numpy()))
                            sample_idx += 1
                            if sample_idx == test_size or True:  # !!!!!!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                return
                loop()
                # MLflow does not have the functionality to log artifacts per training step, so we have to incorporate
                # the training step (iteration_idx) into the artifact path
                mlflow.log_artifacts(temp_dir, 'iteration_%07i' % self.iteration_idx)
                shutil.rmtree(temp_dir)
            self.iteration_idx += 1

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer_or_lr)

    def _fit_model(self, mlflow_run):
        dataset = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                          preprocessing=self.preprocessing)
        self.model.fit(dataset, epochs=self.num_epochs, steps_per_epoch=self.steps_per_training_epoch,
                       callbacks=[TFTrainer.Callback(self, mlflow_run)])

    def _train(self):
        if self.mlflow_experiment_name is not None:
            self._init_mlflow()
            self._compile_model()
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_experiment_name) as run:
                self._fit_model(mlflow_run=run)
        else:
            self._compile_model()
            self._fit_model(mlflow_run=None)
