import abc
import mlflow

from data_handling.dataloader import DataLoader
from utils import *


class Trainer(abc.ABC):
    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, evaluation_interval=None):
        """
        Abstract class for model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
            experiment_name: name of the experiment to log this training run under in MLflow
            run_name: name of the run to log this training run under in MLflow
            split: fraction of dataset provided by the DataLoader which to use for training rather than testing
                   (None to use default)
            num_epochs: number of epochs, i.e. passes through the dataset, to train model for (None to use default)
            batch_size: number of samples to use per training iteration (None to use default)
            optimizer_or_lr: optimizer to use, or learning rate to use with this method's default optimizer
                             (None to use default)
            loss_function: loss function to use (None to use default)
            evaluation_interval: interval, in iterations, in which to perform an evaluation on the test set
                                 (None to use default)
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

    def _init_mlflow(self):
        self.mlflow_experiment_id = None
        if self.mlflow_experiment_name is not None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                self.mlflow_experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
            else:
                self.mlflow_experiment_id = experiment.experiment_id

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
    def train(self):
        """
        Trains the model.
        """
        raise NotImplementedError('Must be defined for trainer.')
