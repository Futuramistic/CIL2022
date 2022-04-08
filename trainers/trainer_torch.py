import abc

import torch
import torch.cuda

from .trainer import Trainer
from utils import *
import mlflow_logger
import numpy as np


class TorchTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None):
        """
        Abstract class for Torch-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, evaluation_interval, num_samples_to_visualize, checkpoint_interval)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.scheduler = scheduler

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
                if self.do_visualize:
                    mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
            self.iteration_idx += 1

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
            preds = torch.argmax(output, dim=1).cpu().numpy()
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
        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing)
        self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                  preprocessing=self.preprocessing)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        model = self.model.to(self.device)
        callback_handler = TorchTrainer.Callback(self, mlflow_run, model)
        for epoch in range(self.num_epochs):
            self._train_step(model, self.device, self.train_loader, callback_handler=callback_handler)
            last_loss = self._eval_step(model, self.device, self.test_loader)
            mlflow_logger.log_metrics({'loss': last_loss})
            if self.do_checkpoint and not epoch % self.checkpoint_interval:
                self._save_checkpoint(model, epoch)
        return last_loss

    def _train_step(self, model, device, train_loader, callback_handler):
        model.train()
        opt = self.optimizer_or_lr
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            y = torch.squeeze(y, dim=1)  # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            preds = model(x)
            print(preds, y)
            loss = self.loss_function(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            del x
            del y
        self.scheduler.step()

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
        print(f'\nTest loss: {test_loss:.3f}')
        return test_loss
    
    def get_F1_score_validation(self, model):
        import losses.f1 as f1
        model.eval()
        f1_scores = []
        for (x,y) in self.test_loader:
            x = x.to(self.device, dtype=torch.float32)
            preds = model(x)
            f1_scores.append(f1.f1_score_torch(preds.detach().cpu().numpy(), y.cpu().numpy()).item())
            del x
            del y
        return torch.mean(f1_scores)