import abc
import mlflow
import pexpect
import requests
import time

from data_handling.dataloader import DataLoader
from utils import *
import mlflow_logger
import numpy as np


class Trainer(abc.ABC):
    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None):
        """
        Abstract class for model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
            experiment_name: name of the experiment to log this training run under in MLflow
            run_name: name of the run to log this training run under in MLflow
            split: fraction of dataset provided by the DataLoader which to use for training rather than test
                   (None to use default)
            num_epochs: number of epochs, i.e. passes through the dataset, to train model for (None to use default)
            batch_size: number of samples to use per training iteration (None to use default)
            optimizer_or_lr: optimizer to use, or learning rate to use with this method's default optimizer
                             (None to use default)
            loss_function: loss function to use (None to use default)
            evaluation_interval: interval, in iterations, in which to perform an evaluation on the test set
                                 (None to use default)
            num_samples_to_visualize: number of samples to visualize predictions for during evaluation
                                      (None to use default)
            checkpoint_interval: interval, in iterations, in which to create model checkpoints
                                 (WARNING: None or 0 to discard model)
        """
        self.dataloader = dataloader
        self.model = model
        self.mlflow_experiment_name = experiment_name
        self.mlflow_experiment_id = None
        self.mlflow_run_name = run_name
        self.split = split
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_or_lr = optimizer_or_lr
        self.loss_function = loss_function
        self.evaluation_interval = evaluation_interval
        self.num_samples_to_visualize = num_samples_to_visualize if num_samples_to_visualize is not None else 6
        self.checkpoint_interval = checkpoint_interval
        self.do_checkpoint = self.checkpoint_interval is not None and self.checkpoint_interval > 0

    def _init_mlflow(self):
        self.mlflow_experiment_id = None
        if self.mlflow_experiment_name is not None:
            def add_known_hosts(host, user, password):
                child = pexpect.spawn('ssh %s@%s' % (user, host))
                i = child.expect(['.* password:', '.* (yes/no)?'])
                if i == 1:
                    child.sendline('yes')
                    child.expect('.* password:')
                child.sendline(password)
                child.expect('.*')
                time.sleep(1)
                child.sendline('exit')

            mlflow_pass = requests.get(MLFLOW_PASS_URL).text
            add_known_hosts(MLFLOW_HOST, MLFLOW_USER, mlflow_pass)

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                self.mlflow_experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
            else:
                self.mlflow_experiment_id = experiment.experiment_id

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        Please add any hyperparameter that you want to be logged to MLFlow
        """
        return {
            'split': self.split,
            'epochs': self.num_epochs,
            'batch size': self.batch_size,
            'optimizer': self.optimizer_or_lr,
            'loss function': self.loss_function,
        }

    @staticmethod
    @abc.abstractmethod
    def get_default_optimizer_with_lr(lr):
        """
        Constructs and returns the default optimizer for this method, with the given learning rate.
        Args:
            lr: the learning rate to use

        Returns: optimizer object (subclass-dependent)
        """
        raise NotImplementedError('Must be defined for trainer.')

    @abc.abstractmethod
    def _fit_model(self, mlflow_run):
        """
        Fit the model.
        """
        raise NotImplementedError('Must be defined for trainer.')

    def train(self):
        """
        Trains the model
        """
        if self.do_checkpoint and not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)
        if self.mlflow_experiment_name is not None:
            self._init_mlflow()
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_experiment_name) as run:
                mlflow_logger.log_hyperparams(self._get_hyperparams())
                mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                last_test_loss = self._fit_model(mlflow_run=run)
                mlflow_logger.log_codebase()
                if self.do_checkpoint:
                    mlflow_logger.log_checkpoints()
        else:
            last_test_loss = self._fit_model(mlflow_run=None)
        return last_test_loss

    @staticmethod
    def _save_image_array(images, directory):

        def segmentation_to_image(x):
            x = (x * 255).astype(int)
            if len(x.shape) < 3:  # if this is true, there are probably bigger problems somewhere else
                x = np.expand_dims(x, axis=0)  # CHW format
            return x

        n = len(images)
        if is_perfect_square(n):
            nb_cols = math.sqrt(n)
        else:
            nb_cols = math.sqrt(next_perfect_square(n))
        nb_cols = int(nb_cols)  # Need it to be an integer
        nb_rows = math.ceil(float(n) / float(nb_cols))  # Number of rows in final image

        # Append enough black images to complete the last non-empty row
        while len(images) < nb_cols * nb_rows:
            images.append(np.zeros_like(images[0]))
        arr = []  # Store images concatenated in the last dimension here
        for i in range(nb_rows):
            row = np.concatenate(images[(i * nb_cols):(i + 1) * nb_cols], axis=-1)
            arr.append(row)
        # Concatenate in the second-to-last dimension to get the final big image
        final = np.concatenate(arr, axis=-2)
        K.preprocessing.image.save_img(os.path.join(directory, f'rgb.png'),
                                       segmentation_to_image(final), data_format="channels_first")