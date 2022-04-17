import abc
import datetime
import numpy as np
import torch
import torch.cuda

from losses.precision_recall_f1 import *
from utils.logging import mlflow_logger
from .trainer import Trainer
from utils import *


class TorchTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None, segmentation_threshold=None):
        """
        Abstract class for Torch-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, evaluation_interval, num_samples_to_visualize, checkpoint_interval,
                         segmentation_threshold)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.scheduler = scheduler

    # This class is a mimicry of TensorFlow's "Callback" class behavior, used not out of necessity (as we write the
    # training loop explicitly in Torch, instead of using the "all-inclusive" model.fit(...) in TF, and can thus just
    # build event handlers into the training loop), but to have a more consistent codebase across TorchTrainer and
    # TFTrainer.
    # See the comment next to the KC.Callback subclass in "trainer_tf.py" for more information.
    class Callback:
        def __init__(self, trainer, mlflow_run, model):
            super().__init__()
            self.model = model
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_batch_end(self):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                precision, recall, f1_score = self.trainer.get_precision_recall_F1_score_validation()
                metrics = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
                print('Metrics at iteration %i: %s' % (self.iteration_idx, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics)
                    if self.do_visualize:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
            self.iteration_idx += 1

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, directory):
        # Sample image indices to visualize
        length = len(self.test_loader)
        indices = random.sample(range(length), self.num_samples_to_visualize)
        images = []

        for i, (batch_xs, batch_ys) in enumerate(self.test_loader):
            if i not in indices:  # TODO works but dirty, find a better solution..
                continue
            batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.numpy()
            output = self.model(batch_xs)
            preds = (output >= self.segmentation_threshold).float().cpu().detach().numpy()
            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            self._fill_images_array(preds, batch_ys, images)

        self._save_image_array(images, directory)

    def _save_checkpoint(self, model, epoch):
        checkpoint_name = f'{CHECKPOINTS_DIR}/cp_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer_or_lr.state_dict()
        }, checkpoint_name)

    def _fit_model(self, mlflow_run):
        print('\nTraining started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        print('Hyperparameters:')
        print(self._get_hyperparams())
        print('')

        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing)
        self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                  preprocessing=self.preprocessing)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}\n')
        model = self.model.to(self.device)

        callback_handler = TorchTrainer.Callback(self, mlflow_run, model)
        last_checkpoint_epoch = -1
        for epoch in range(self.num_epochs):
            last_train_loss = self._train_step(model, self.device, self.train_loader, callback_handler=callback_handler)
            last_test_loss = self._eval_step(model, self.device, self.test_loader)
            metrics = {'train_loss': last_train_loss, 'test_loss': last_test_loss}

            print('\nEpoch %i finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) % epoch)
            print('Metrics: %s\n' % str(metrics))

            mlflow_logger.log_metrics(metrics)
            
            if self.do_checkpoint and not epoch % self.checkpoint_interval:
                self._save_checkpoint(model, epoch)
                last_checkpoint_epoch = epoch
                
            mlflow_logger.log_logfiles()
        if self.do_checkpoint and last_checkpoint_epoch < epoch:
            # save final checkpoint
            self._save_checkpoint(model, epoch)
        
        print('\nTraining finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        mlflow_logger.log_logfiles()
        return last_test_loss

    def _train_step(self, model, device, train_loader, callback_handler):
        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0
        for (x, y) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            y = torch.squeeze(y, dim=1)  # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            preds = model(x)
            loss = self.loss_function(preds, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            del x
            del y
        train_loss /= len(train_loader.dataset)
        self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
                y = torch.squeeze(y, dim=1)
                preds = model(x)
                test_loss += self.loss_function(preds, y).item()
                del x
                del y
        test_loss /= len(test_loader.dataset)
        return test_loss

    def get_F1_score_validation(self):
        _, _, f1_score = self.get_precision_recall_F1_score_validation()
        return f1_score

    def get_precision_recall_F1_score_validation(self):
        self.model.eval()
        precisions, recalls, f1_scores = [], [], []
        for (x, y) in self.test_loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            output = self.model(x)
            preds = (output >= self.segmentation_threshold).float()
            precision, recall, f1_score = precision_recall_f1_score_torch(preds, y)
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1_scores.append(f1_score.cpu().numpy())
            del x
            del y
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
