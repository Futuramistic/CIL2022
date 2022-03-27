import abc
import mlflow
import pexpect
import requests
import time

from data_handling.dataloader import DataLoader
from utils import *


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

        if checkpoint_interval is not None and checkpoint_interval > 0:
            raise NotImplementedError()

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
