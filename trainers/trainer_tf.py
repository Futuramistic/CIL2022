import abc
import datetime
import hashlib
# from typing import final
import numpy as np
import os
import pysftp
import requests
import shutil
import tempfile
import tensorflow as tf
import tensorflow.keras.callbacks as KC
from urllib.parse import urlparse
from blobs_remover import remove_blobs
from threshold_optimizer import ThresholdOptimizer

from requests.auth import HTTPBasicAuth

from losses.loss_harmonizer import DEFAULT_TF_DIM_LAYOUT
from losses.precision_recall_f1 import *
from utils.logging import mlflow_logger
from .trainer import Trainer
from utils import *


class TFTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing, steps_per_training_epoch,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None, load_checkpoint_path=None,
                 segmentation_threshold=None, use_channelwise_norm=False, blobs_removal_threshold=0, hyper_seg_threshold=False):
        """
        Abstract class for TensorFlow-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, loss_function_hyperparams, evaluation_interval, num_samples_to_visualize,
                         checkpoint_interval, load_checkpoint_path, segmentation_threshold, use_channelwise_norm,
                         blobs_removal_threshold, hyper_seg_threshold)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.steps_per_training_epoch = steps_per_training_epoch

        if hyper_seg_threshold:
            self.seg_thresh_dataloader = self.dataloader.get_training_dataloader(split=0.2, batch_size=1,
                                                                preprocessing=self.preprocessing)

            # initialize again, else we will not use entire training dataset but only a split of 0.2 !
            self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size, preprocessing=self.preprocessing)
        
    # Subclassing tensorflow.keras.callbacks.Callback (here: KC.Callback) allows us to override various functions to be
    # called when specific events occur while fitting a model using TF's model.fit(...). An instance of the subclass
    # needs to be passed in the "callbacks" parameter (which, if specified, can either be a single instance, or a list
    # of instances of KC.Callback subclasses)
    class Callback(KC.Callback):
        def __init__(self, trainer, mlflow_run):
            super().__init__()
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.epoch_iteration_idx = 0
            self.epoch_idx = 0
            self.best_score = -1
            self.best_val_loss = 1e5
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_begin(self, logs=None):
            print('\nTraining started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print(f'Session ID: {SESSION_ID}')
            print('Hyperparameters:')
            print(self.trainer._get_hyperparams())
            print('')

        def on_train_end(self, logs=None):
            print('\nTraining finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            mlflow_logger.log_logfiles()

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs=None):
            print('\n\nEpoch %i finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) % epoch)
            print('Metrics: %s\n' % str(logs))
            mlflow_logger.log_metrics(logs, aggregate_iteration_idx=self.iteration_idx)
            mlflow_logger.log_logfiles()
            
            # checkpoints should be logged to MLflow right after their creation, so that if training is
            # stopped/crashes *without* reaching the final "mlflow_logger.log_checkpoints()" call in trainer.py,
            # prior checkpoints have already been persisted
            # since we don't have a way of getting notified when KC.ModelCheckpoint has finished creating the checkpoint,
            # we simply check at the end of each epoch whether there are any checkpoints to upload and upload them
            # if necessary
            
            if self.trainer.do_checkpoint and self.best_val_loss > logs['val_loss']:
                self.best_val_loss = logs['val_loss']
                keras.models.save_model(model=self.model,filepath=os.path.join(CHECKPOINTS_DIR, "cp_best_val_loss.ckpt"))

            mlflow_logger.log_checkpoints()
            
            self.epoch_idx += 1
            self.epoch_iteration_idx = 0

            # it seems we can only safely delete the original checkpoint dir after having trained for at least one
            # iteration
            # if self.original_checkpoint_hash is not None and os.path.isdir(f'original_checkpoint_{self.original_checkpoint_hash}.ckpt'):
            #     shutil.rmtree(f'original_checkpoint_{self.original_checkpoint_hash}.ckpt')


        def on_train_batch_begin(self, batch, logs=None):
            pass

        def on_train_batch_end(self, batch, logs=None):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                precision, recall, f1_score, self.segmentation_threshold = self.trainer.get_precision_recall_F1_score_validation()
                metrics = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'seg_threshold': self.segmentation_threshold}
                print('\nMetrics at aggregate iteration %i (ep. %i, ep.-it. %i): %s'
                      % (self.iteration_idx, self.epoch_idx, batch, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=self.iteration_idx)
                    if self.do_visualize:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx,self.epoch_idx,self.epoch_iteration_idx)
                # save the best f1 score checkpoint
                if self.trainer.do_checkpoint and self.best_score <= f1_score:
                    self.best_score = f1_score
                    keras.models.save_model(model=self.model,filepath=os.path.join(CHECKPOINTS_DIR, "cp_best_f1.ckpt"))

            if self.trainer.do_checkpoint\
                and self.iteration_idx % self.trainer.checkpoint_interval == 0\
                and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                checkpoint_path = f'{CHECKPOINTS_DIR}/cp_ep-{"%05i" % self.epoch_idx}'+\
                                  f'_it-{"%05i" % self.epoch_iteration_idx}' +\
                                  f'_step-{self.iteration_idx}.ckpt'
                keras.models.save_model(model=self.trainer.model, filepath=checkpoint_path)
            
            self.iteration_idx += 1
            self.epoch_iteration_idx += 1

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path, iteration_index, epoch_idx, epoch_iteration_idx):
        # for batch_xs, batch_ys in self.test_loader.shuffle(10 * num_samples).batch(num_samples):

        # fix half of the samples, randomize other half
        # the first, fixed half of samples serves for comparison purposes across models/runs
        # the second, randomized half allows us to spot weaknesses of the model we might miss when
        # always visualizing the same samples

        num_to_visualize = self.num_samples_to_visualize
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)

        _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)
        if num_to_visualize >= test_dataset_size:
            # just visualize the entire test set
            vis_dataloader = self.test_loader.take(test_dataset_size).batch(vis_batch_size)
        else:
            num_fixed_samples = num_to_visualize // 2
            num_random_samples = num_to_visualize - num_fixed_samples

            fixed_samples = self.test_loader.take(num_fixed_samples)
            random_samples = self.test_loader.skip(num_fixed_samples).take(num_random_samples).shuffle(num_random_samples)
            vis_dataloader = fixed_samples.concatenate(random_samples).batch(vis_batch_size)

        images = []

        for (batch_xs, batch_ys) in vis_dataloader:
            batch_xs = tf.squeeze(batch_xs, axis=1)
            batch_ys = tf.squeeze(batch_ys, axis=1).numpy()
            output = self.model.predict(batch_xs)  # returns np.ndarray

            # More channels than needed - U^2-Net-style
            if(len(output.shape)==5):
                output = output[0]

            channel_dim_idx = DEFAULT_TF_DIM_LAYOUT.find('C')
            if output.shape[channel_dim_idx] > 1:
                output = np.argmax(output, axis=channel_dim_idx)
                output = np.expand_dims(output, axis=channel_dim_idx)

            preds = (output >= self.segmentation_threshold).astype(np.float)
            batch_ys = np.moveaxis(batch_ys, -1, 1)  # TODO only do this if we know the network uses HWC format
            preds = np.moveaxis(preds, -1, 1)

            # preds = remove_blobs(preds, self.blobs_removal_threshold)

            # print('shape', preds.shape)
            preds_list = []
            for i in range(preds.shape[0]):
                # print('preds[i]', preds[i].shape)
                pred_ = remove_blobs(preds[i], threshold=self.blobs_removal_threshold)
                # print('pred_', pred_.shape)
                if len(pred_.shape) == 2:
                    preds_list.append(pred_[None, None, :, :])
                elif len(pred_.shape) == 3:
                    preds_list.append(pred_[None, :, :, :])
                else:
                    print('problem', pred_.shape)
            # print('len', len(preds_list))
            preds = np.concatenate(preds_list, axis=0)
            # print(preds.shape)

            # preds = np.expand_dims(preds, axis=1)  # so add it back, in CHW format

            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            self._fill_images_array(preds, batch_ys, images)

        self._save_image_array(images, file_path)

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer_or_lr)

    def _load_checkpoint(self, checkpoint_path):
        print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
        load_from_sftp = checkpoint_path.lower().startswith('sftp://')
        if load_from_sftp:
            self.original_checkpoint_hash = hashlib.md5(str.encode(checkpoint_path)).hexdigest()
            # in TF, even though the checkpoint names all end in ".ckpt", they are actually directories
            # hence we have to use sftp_download_dir_portable to download them
            final_checkpoint_path = f'original_checkpoint_{self.original_checkpoint_hash}.ckpt'
            if not os.path.isdir(final_checkpoint_path):
                os.makedirs(final_checkpoint_path, exist_ok=True)
                print(f'Downloading checkpoint from "{checkpoint_path}" to "{final_checkpoint_path}"...')
                cnopts = pysftp.CnOpts()
                cnopts.hostkeys = None
                mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL,
                                            auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'],
                                                                os.environ['MLFLOW_TRACKING_PASSWORD'])).text
                url_components = urlparse(checkpoint_path)
                with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_FTP_USER, password=mlflow_ftp_pass,
                                    cnopts=cnopts) as sftp:
                    sftp_download_dir_portable(sftp, remote_dir=url_components.path, local_dir=final_checkpoint_path)
                print(f'Download successful')
            else:
                print(f'Checkpoint "{checkpoint_path}", to be downloaded to "{final_checkpoint_path}", found on disk')
        else:
            final_checkpoint_path = checkpoint_path

        print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
        self.model.load_weights(final_checkpoint_path)
        print('Checkpoint loaded\n')

        # Note that the final_checkpoint_path directory cannot be deleted right away! This leads to errors.
        # As a workaround, we delete the directory after the end of the first epoch.


    def _fit_model(self, mlflow_run):
        self._compile_model()
        
        if self.load_checkpoint_path is not None:
            self._load_checkpoint(self.load_checkpoint_path)
        
        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing)
        self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                  preprocessing=self.preprocessing)
        _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)

        callbacks = [TFTrainer.Callback(self, mlflow_run)]
        # model checkpointing functionality moved into TFTrainer.Callback to allow for custom checkpoint names
        
        self.model.fit(self.train_loader, validation_data=self.test_loader.take(test_dataset_size), epochs=self.num_epochs,
                       steps_per_epoch=self.steps_per_training_epoch, callbacks=callbacks, verbose=1 if IS_DEBUG else 2)
        
        if self.do_checkpoint:
            # save final checkpoint
            keras.models.save_model(model=self.model,
                                    filepath=os.path.join(CHECKPOINTS_DIR, "cp_final.ckpt"))

    def get_F1_score_validation(self):
        _, _, f1_score, _ = self.get_precision_recall_F1_score_validation()
        return f1_score

    def get_precision_recall_F1_score_validation(self):
        precisions, recalls, f1_scores = [], [], []
        threshold = self.segmentation_threshold
        if self.hyper_seg_threshold:
            threshold = self.get_best_segmentation_threshold()
        _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)
        for x, y in self.test_loader.take(test_dataset_size):
            output = self.model(x)
            
            # More channels than needed - U^2-Net-style
            if(len(output.shape)==5):
                output = output[0]
            preds = tf.cast(tf.squeeze(output) >= threshold, tf.dtypes.int8)
            # print('tf preds', preds.shape)
            blb_input = preds.numpy()
            preds = remove_blobs(blb_input, threshold=self.blobs_removal_threshold)
            # print('tf preds 2', preds.shape)
            precision, recall, f1_score = precision_recall_f1_score_tf(preds, y)
            precisions.append(precision.numpy().item())
            recalls.append(recall.numpy().item())
            f1_scores.append(f1_score.numpy().item())
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), threshold

    ''' def find_best_segmentation_threshold(self,step=0.05):
        # Save original threshold
        original_threshold = self.segmentation_threshold
        best_threshold = None
        best_f1 = 0.0
        for i in np.arange(step,1+step,step):
            self.segmentation_threshold = i
            _,_,f1 = self.get_precision_recall_F1_score_validation()
            if(best_f1<=f1):
                best_f1 = f1
                best_threshold = self.segmentation_threshold
        # Restore to the original threshold
        self.segmentation_threshold = original_threshold
        print(f'F1 score: {best_f1:.4f} for threshold: {best_threshold:.4f}')
        return best_threshold '''
    
    def get_best_segmentation_threshold(self):
        predictions = []
        targets = []
        threshold_optimizer = ThresholdOptimizer()
        dataloader_len, _, _ = self.dataloader.get_dataset_sizes(split=0.2)
        for (x,y) in self.seg_thresh_dataloader.take(dataloader_len):
            output = self.model(x)
            # More channels than needed - U^2-Net-style
            if(len(output.shape)==5):
                output = output[0]
            blb_input = tf.squeeze(output).numpy()
            preds = remove_blobs(blb_input, threshold=self.blobs_removal_threshold)
            predictions.append(preds)
            targets.append(y)
        best_threshold = threshold_optimizer.run(predictions, targets, f1_score_tf)
        return best_threshold
