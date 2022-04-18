import abc
import tensorflow.keras as K
import mlflow
import numpy as np
import os
import pexpect
import paramiko
import pysftp
import requests
import shutil
import socket
import time

from data_handling import DataLoader
from utils.logging import mlflow_logger, optim_hyparam_serializer
from utils import *


class Trainer(abc.ABC):
    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None, load_checkpoint_path=None,
                 segmentation_threshold=None):
        """
        Abstract class for model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
            experiment_name: name of the experiment to log this training run under in MLflow
            run_name: name of the run to log this training run under in MLflow (None to use default name assigned by
                      MLflow)
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
                                 specify an extremely high number (e.g. 1e15) to only create a single checkpoint after training has finished
                                 (WARNING: None or 0 to discard model)
            load_checkpoint_path: path to checkpoint file, or SFTP checkpoint URL for MLflow, to load a checkpoint and
                                  resume training from (None to start training from scratch instead)
            segmentation_threshold: threshold >= which to consider the model's prediction for a given pixel to
                                    correspond to class 1 rather than class 0 (None to use default)
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
        self.segmentation_threshold =\
            segmentation_threshold if segmentation_threshold is not None else DEFAULT_SEGMENTATION_THRESHOLD
        self.is_windows = os.name == 'nt'
        self.load_checkpoint_path = load_checkpoint_path
        if not self.do_checkpoint:
            print('\n*** WARNING: no checkpoints of this model will be created! Specify valid checkpoint_interval '
                  '(in iterations) to Trainer in order to create checkpoints. ***\n')

    def _init_mlflow(self):
        self.mlflow_experiment_id = None
        if self.mlflow_experiment_name is not None:
            def add_known_hosts(host, user, password, jump_host=None):
                spawn_str =\
                    'ssh %s@%s' % (user, host) if jump_host is None else 'ssh -J %s %s@%s' % (jump_host, user, host)
                if self.is_windows:
                    # pexpect.spawn not supported on windows
                    import wexpect
                    child = wexpect.spawn(spawn_str)
                else:
                    child = pexpect.spawn(spawn_str)
                i = child.expect(['.*ssword.*', '.*(yes/no).*'])
                if i == 1:
                    child.sendline('yes')
                    child.expect('.*ssword.*')
                child.sendline(password)
                child.expect('.*')
                time.sleep(1)
                child.sendline('exit')

                if jump_host is not None:
                    if self.is_windows:
                        raise RuntimeError('Use of jump hosts for Trainer not supported on Windows machines')

                    # monkey-patch pysftp to use the provided jump host

                    def new_start_transport(self, host, port):
                        try:
                            jumpbox = paramiko.SSHClient()
                            jumpbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            jumpbox.connect(jump_host,
                                            key_filename=os.path.join(*[os.getenv('HOME'), '.ssh', 'id_' + jump_host]))

                            jumpbox_transport = jumpbox.get_transport()
                            dest_addr = (host, port)
                            jumpbox_channel = jumpbox_transport.open_channel('direct-tcpip', dest_addr, ('', 0))

                            target = paramiko.SSHClient()
                            target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            target.connect(host, port, user, password, sock=jumpbox_channel)

                            self._transport = target.get_transport()
                            self._transport.connect = lambda *args, **kwargs: None  # ignore subsequent "connect" calls

                            # set security ciphers if set
                            if self._cnopts.ciphers is not None:
                                ciphers = self._cnopts.ciphers
                                self._transport.get_security_options().ciphers = ciphers
                        except (AttributeError, socket.gaierror):
                            # couldn't connect
                            raise pysftp.ConnectionException(host, port)

                    pysftp.Connection._start_transport = new_start_transport

            mlflow_init_successful = True
            MLFLOW_INIT_ERROR_MSG = 'MLflow initialization failed. Will not use MLflow for this run.'

            try:
                mlflow_pass = requests.get(MLFLOW_PASS_URL).text
                try:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_USER, mlflow_pass)
                except:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_USER, mlflow_pass, MLFLOW_JUMP_HOST)
            except:
                mlflow_init_successful = False
                print(MLFLOW_INIT_ERROR_MSG)

            if mlflow_init_successful:
                try:
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                    if experiment is None:
                        self.mlflow_experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
                    else:
                        self.mlflow_experiment_id = experiment.experiment_id
                except:
                    mlflow_init_successful = False
                    print(MLFLOW_INIT_ERROR_MSG)

            return mlflow_init_successful
        else:
            return False

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        Please add any hyperparameter that you want to be logged to MLFlow
        """
        return {
            'split': self.split,
            'epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'loss_function': self.loss_function,
            'seg_threshold': self.segmentation_threshold,
            'model': self.model.name if hasattr(self.model, 'name') else type(self.model).__name__,
            'dataset': self.dataloader.dataset,
            'from_checkpoint': self.load_checkpoint_path if self.load_checkpoint_path is not None else '',
            'session_id': SESSION_ID,
            **(optim_hyparam_serializer.serialize_optimizer_hyperparams(self.optimizer_or_lr))
        }

    @staticmethod
    @abc.abstractmethod
    def get_default_optimizer_with_lr(lr, model):
        """
        Constructs and returns the default optimizer for this method, with the given learning rate and model.
        Args:
            lr: the learning rate to use
            model: the model to use

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
        
        if self.mlflow_experiment_name is not None and self._init_mlflow():
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_run_name) as run:
                try:
                    mlflow_logger.log_hyperparams(self._get_hyperparams())
                    mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                    last_test_loss = self._fit_model(mlflow_run=run)
                    mlflow_logger.log_codebase()
                    if self.do_checkpoint:
                        mlflow_logger.log_checkpoints()
                    mlflow_logger.log_logfiles()
                except Exception as e:
                    print(f'\n\n*** Exception encountered: ***\n{e}\n')
                    mlflow_logger.log_logfiles()
                    raise e
        else:
            last_test_loss = self._fit_model(mlflow_run=None)

        if os.path.exists(CHECKPOINTS_DIR):
            shutil.rmtree(CHECKPOINTS_DIR)
            
        return last_test_loss

    @staticmethod
    def _fill_images_array(preds, batch_ys, images):
        if batch_ys is None:
            batch_ys = np.zeros_like(preds)
        merged = (2 * preds + batch_ys)
        green_channel = merged == 3  # true positives
        red_channel = merged == 2  # false positive
        blue_channel = merged == 1  # false negative
        rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=1)
        for batch_sample_idx in range(preds.shape[0]):
            images.append(rgb[batch_sample_idx])

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
