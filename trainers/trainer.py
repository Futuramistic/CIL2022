import abc
import inspect
import mlflow
import numpy as np
import pexpect
import paramiko
import pysftp
import requests
import shutil
import socket

from losses import *
from requests.auth import HTTPBasicAuth
from utils import *
from utils.logging import mlflow_logger, optim_hyparam_serializer
import tensorflow.keras as K


class Trainer(abc.ABC):
    """
    Abstract class for model trainers.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, use_channelwise_norm=False,
                 blobs_removal_threshold=0, hyper_seg_threshold=False, use_sample_weighting=False,
                 use_adaboost=False, f1_threshold_to_log_checkpoint=DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT):
        """
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
            loss_function: (name of) loss function to use (None to use default)
            loss_function_hyperparams: hyperparameters of loss function to use
                                       (will be bound to the loss function automatically; None to skip)
            evaluation_interval: interval, in iterations, in which to perform an evaluation on the test set
                                 (None to use default)
            num_samples_to_visualize: number of samples to visualize predictions for during evaluation
                                      (None to use default)
            checkpoint_interval: interval, in iterations, in which to create model checkpoints
                                 specify an extremely high number (e.g. 1e15) to only create a single checkpoint after
                                 training has finished (WARNING: None or 0 to discard model)
            load_checkpoint_path: path to checkpoint file, or SFTP checkpoint URL for MLflow, to load a checkpoint and
                                  resume training from (None to start training from scratch instead)
            segmentation_threshold: threshold >= which to consider the model's prediction for a given pixel to
                                    correspond to class 1 rather than class 0 (None to use default)
            hyper_seg_threshold: whether to use hyperopt to calculate the optimal threshold on the evaluation data 
                                 (measured by F1 score)
            use_sample_weighting: whether to use sample weighting to train more on samples with worse losses; weights 
                                 are recalculated after each epoch
            use_adaboost: If True, the trainer is part of the adaboost algorithm
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
        self.loss_function_hyperparams = loss_function_hyperparams if loss_function_hyperparams is not None else {}
        self.hyper_seg_threshold = hyper_seg_threshold
        self.use_sample_weighting = use_sample_weighting
        self.f1_threshold_to_log_checkpoint = f1_threshold_to_log_checkpoint

        self.loss_function_name = str(loss_function)
        if isinstance(loss_function, str):
            # additional imports to expand scope of losses accessible via "eval"
            import torch
            import torch.nn
            import tensorflow.keras.losses
            self.loss_function = eval(loss_function)
        else:
            self.loss_function = loss_function
        if inspect.isclass(self.loss_function):
            # instantiate class with given hyperparameters
            self.loss_function = self.loss_function(**self.loss_function_hyperparams)
        elif inspect.isfunction(self.loss_function):
            self.orig_loss_function = self.loss_function
            self.loss_function = lambda *args, **kwargs: self.orig_loss_function(*args, **kwargs,
                                                                                 **self.loss_function_hyperparams)

        self.evaluation_interval = evaluation_interval
        self.num_samples_to_visualize =\
            num_samples_to_visualize if num_samples_to_visualize is not None else DEFAULT_NUM_SAMPLES_TO_VISUALIZE
        self.checkpoint_interval = checkpoint_interval
        self.do_checkpoint = self.checkpoint_interval is not None and self.checkpoint_interval > 0
        self.segmentation_threshold =\
            segmentation_threshold if segmentation_threshold is not None else DEFAULT_SEGMENTATION_THRESHOLD
        self.use_channelwise_norm = use_channelwise_norm
        self.blobs_removal_threshold = blobs_removal_threshold
        self.is_windows = os.name == 'nt'
        self.load_checkpoint_path = load_checkpoint_path
        self.mlflow_initialized = False
        self.original_checkpoint_hash = None
        if not self.do_checkpoint:
            print('\n*** WARNING: no checkpoints of this model will be created! Specify valid checkpoint_interval '
                  '(in iterations) to Trainer in order to create checkpoints. ***\n')
        
        self.adaboost = use_adaboost
        if self.adaboost:
            self.curr_best_checkpoint_path = None
            self.checkpoints_folder = None

    def _init_mlflow(self):
        """
        Initialize a connection to the MLFlow server
        Returns:
            True if successfully established a connection
        """
        # return False # TODO: revert after debug
        if self.mlflow_initialized:
            return True
        
        self.mlflow_experiment_id = None
        if self.mlflow_experiment_name not in ['', None]:
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
                os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_HTTP_USER
                os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_HTTP_PASS

                mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL,
                                               auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'],
                                                                  os.environ['MLFLOW_TRACKING_PASSWORD'])).text
                try:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_FTP_USER, mlflow_ftp_pass)
                except:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_FTP_USER, mlflow_ftp_pass, MLFLOW_JUMP_HOST)
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
                    self.mlflow_initialized = True
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
            'seg_threshold': self.segmentation_threshold,
            'loss_function': getattr(self, 'loss_function_name', self.loss_function),
            'use_sample_weighting': getattr(self, 'use_sample_weighting', False),
            'use_channelwise_norm': getattr(self, 'use_channelwise_norm', False),
            'blobs_removal_threshold': getattr(self, 'blobs_removal_threshold', 0),
            'model': getattr(self.model, 'name', type(self.model).__name__),
            'dataset': self.dataloader.dataset,
            'dl_use_geometric_augmentation': getattr(self.dataloader, 'use_geometric_augmentation', False),
            'dl_use_color_augmentation': getattr(self.dataloader, 'use_color_augmentation', False),
            'dl_aug_contrast': getattr(self.dataloader, 'contrast', None),
            'dl_aug_brightness': getattr(self.dataloader, 'brightness', None),
            'dl_aug_saturation': getattr(self.dataloader, 'saturation', None),
            'from_checkpoint': self.load_checkpoint_path if self.load_checkpoint_path is not None else '',
            'session_id': SESSION_ID,
            'use_hyperopt_for_optimal_threshold': self.hyper_seg_threshold,
            'use_sample_weighting': self.use_sample_weighting,
            'use_adaboost': self.adaboost,
            **(optim_hyparam_serializer.serialize_optimizer_hyperparams(self.optimizer_or_lr)),
            **({f'loss_{k}': v for k, v in self.loss_function_hyperparams.items()})
        }

    @staticmethod
    @abc.abstractmethod
    def get_default_optimizer_with_lr(lr, model):
        """
        Construct and return the default optimizer for this method, with the given learning rate and model.
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

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        """
        raise NotImplementedError('Must be defined for trainer.')

    @abc.abstractmethod
    def get_precision_recall_F1_score_validation(self):
        """
        Calculate and return the precision, recall and F1 score on the validation split of the current dataset, as well
        as the segmentation threshold used to calculate these metrics.
        Returns: Tuple of (float, float, float, float) containing (precision, recall, f1_score, segmentation_threshold)
        """
        raise NotImplementedError('Must be defined for trainer.')

    def train(self):
        """
        Trains the model
        """
        if self.do_checkpoint and not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)
        
        if self.mlflow_experiment_name not in ['', None] and self._init_mlflow():
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_run_name) as run:
                try:
                    mlflow_logger.log_hyperparams(self._get_hyperparams())
                    mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                    mlflow_logger.log_codebase()  # log codebase before training, to be invariant to train crashes/stops
                    mlflow_logger.log_command_line()  # log command line used to execute the script, if available
                    last_test_loss = self._fit_model(mlflow_run=run)
                    if self.do_checkpoint:
                        remove_local_checkpoint = not self.adaboost
                        other_checkpoint_name = None if not self.adaboost else self.checkpoints_folder
                        mlflow_logger.log_checkpoints(remove_local_checkpoint, other_checkpoint_name)
                    mlflow_logger.log_logfiles()
                except Exception as e:
                    err_msg = f'*** Exception encountered: ***\n{e}'
                    print(f'\n\n{err_msg}\n')
                    mlflow_logger.log_logfiles()
                    if not IS_DEBUG:
                        pushbullet_logger.send_pushbullet_message(err_msg)
                    raise e
        else:
            last_test_loss = self._fit_model(mlflow_run=None)

        if os.path.exists(CHECKPOINTS_DIR) and not self.adaboost:
            shutil.rmtree(CHECKPOINTS_DIR)

        return last_test_loss

    def eval(self):
        """
        Evaluate the model on the validation split of the current dataset, using the precision, recall
        and F1 score metrics
        """
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            # initialize by calling get_training_dataloader
            self.dataloader.get_training_dataloader(split=self.split, batch_size=1, preprocessing=self.preprocessing)
            self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                      preprocessing=self.preprocessing)

        if self.mlflow_experiment_name not in ['', None] and self._init_mlflow():
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_run_name) as run:
                try:
                    mlflow_logger.log_hyperparams(self._get_hyperparams())
                    mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                    mlflow_logger.log_codebase()  # log codebase before training, to be invariant to train crashes/stops
                    mlflow_logger.log_command_line()  # log command line used to execute the script, if available
                    
                    precisions_road, recalls_road, f1_road_scores, precisions_bkgd, \
                        recalls_bkgd, f1_bkgd_scores, f1_macro_scores, f1_weighted_scores,\
                            f1_road_patchified_scores, f1_bkgd_patchified_scores, f1_patchified_weighted_scores,\
                                threshold = self.get_precision_recall_F1_score_validation()
                    metrics = {'precisions_road': precisions_road, 'recalls_road': recalls_road, 'f1_road_scores': f1_road_scores,
                               'precisions_bkgd': precisions_bkgd, 'recalls_bkgd': recalls_bkgd, 'f1_bkgd_scores': f1_bkgd_scores,
                               'f1_macro_scores': f1_macro_scores, 'f1_weighted_scores': f1_weighted_scores, 
                               'f1_road_patchified_scores': f1_road_patchified_scores, 'f1_bkgd_patchified_scores':f1_bkgd_patchified_scores,
                               'f1_patchified_weighted_scores':f1_patchified_weighted_scores, 'seg_threshold': threshold}
                    print(f'Evaluation metrics: {metrics}')
                    if mlflow_logger.logging_to_mlflow_enabled():
                        mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=0)
                        if self.num_samples_to_visualize is not None and self.num_samples_to_visualize > 0:
                            mlflow_logger.log_visualizations(self, 0, 0, 0)

                    mlflow_logger.log_logfiles()
                except Exception as e:
                    err_msg = f'*** Exception encountered: ***\n{e}'
                    print(f'\n\n{err_msg}\n')
                    mlflow_logger.log_logfiles()
                    if not IS_DEBUG:
                        pushbullet_logger.send_pushbullet_message(err_msg)
                    raise e
        else:
            precisions_road, recalls_road, f1_road_scores, precisions_bkgd, \
                        recalls_bkgd, f1_bkgd_scores, f1_macro_scores, f1_weighted_scores,\
                            f1_road_patchified_scores, f1_bkgd_patchified_scores, f1_patchified_weighted_scores,\
                                threshold = self.get_precision_recall_F1_score_validation()
            metrics = {'precisions_road': precisions_road, 'recalls_road': recalls_road, 'f1_road_scores': f1_road_scores,
                               'precisions_bkgd': precisions_bkgd, 'recalls_bkgd': recalls_bkgd, 'f1_bkgd_scores': f1_bkgd_scores,
                               'f1_macro_scores': f1_macro_scores, 'f1_weighted_scores': f1_weighted_scores, 
                               'f1_road_patchified_scores': f1_road_patchified_scores, 'f1_bkgd_patchified_scores':f1_bkgd_patchified_scores,
                               'f1_patchified_weighted_scores':f1_patchified_weighted_scores, 'seg_threshold': threshold}
            print(f'Evaluation metrics: {metrics}')

        return metrics

    def create_visualizations(self, vis_file_path, iteration_index, epoch_idx, epoch_iteration_idx):
        """
        Create visualizations for the validation dataset and save them into "vis_file_path".
        """
        raise NotImplementedError('Must be defined for trainer.')
    
    def get_best_segmentation_threshold(self):
        """Uses Hyperopt to get the segmentation threshold with the best F1 score based on a subset of the training data
        Returns:
            float: best segmentation threshold of the current model
        """
        raise NotImplementedError('Must be defined for trainer')

    @staticmethod
    def _fill_images_array(preds, batch_ys, images):
        """
        Create images that visualize TP/TN/FP/FN statistics and add them to the given 'images' list
        Args:
            preds (np.ndarray): batch of images predicted by some model
            batch_ys (np.ndarray): batch of groundtruth images
            images (list): List to append the visualizations to
        """
        if batch_ys is None:
            batch_ys = np.zeros_like(preds)
        if len(batch_ys.shape) > len(preds.shape):
            # collapse channel dimension
            batch_ys = np.argmax(batch_ys, axis=-1)
        merged = (2 * preds + batch_ys)
        green_channel = merged == 3  # true positives
        red_channel = merged == 2  # false positive
        blue_channel = merged == 1  # false negative
        rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=1)
        for batch_sample_idx in range(preds.shape[0]):
            images.append(rgb[batch_sample_idx])

    @staticmethod
    def _save_image_array(images, file_path):
        """
        Given a list of visualization images, create one big image that concatenates all of them in an
        optimal format
        """
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
        K.utils.save_img(file_path, segmentation_to_image(final), data_format="channels_first")