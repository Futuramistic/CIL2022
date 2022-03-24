import abc
import mlflow
import numpy as np
import shutil
import tempfile
import time

import torch
import torch.cuda
from torchvision.utils import save_image

from trainer import Trainer
from utils import *

###########################
####### NOT TESTED ########
###########################
class TorchTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing, steps_per_training_epoch,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, loss_function=None, evaluation_interval=None,
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
        self.steps_per_training_epoch = steps_per_training_epoch

    class Callback:
        def __init__(self, trainer, mlflow_run, model):
            super().__init__()
            self.model = model
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.testing_dl = self.trainer.dataloader.get_testing_dataloader(split=self.trainer.split, batch_size=1,
                                                                             preprocessing=self.trainer.preprocessing)
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_batch_end(self):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                eval_start = time.time()
                # remember: dataloader may be infinite

                # store segmentations in temp_dir, then upload temp_dir to Mlflow server
                if self.do_visualize:
                    temp_dir = tempfile.mkdtemp()

                def segmentation_to_image(x):
                    x = (x * 255).astype(int)
                    if len(x.shape) < 3:
                        x = np.expand_dims(x, axis=-1)
                    return x

                sample_idx = 0

                def loop():
                    nonlocal sample_idx
                    if self.do_visualize:
                        for batch_xs, batch_ys in self.testing_dl.take(self.trainer.num_samples_to_visualize):
                            for batch_sample_idx in range(batch_xs.shape[0]):
                                logits = self.model.predict(batch_xs)
                                preds = torch.argmax(logits, dim=-1)
                                # TODO: merge prediction and ground truth into one image using colors, to reduce traffic
                                # TODO: merge multiple images to one big figure to reduce number of requests
                                save_image(preds[batch_sample_idx], os.path.join(temp_dir, f'{sample_idx}_pred.png'))
                                if batch_ys is not None:
                                    save_image(batch_ys[batch_sample_idx], os.path.join(temp_dir, f'{sample_idx}_gt.png'))
                                sample_idx += 1

                eval_inference_start = time.time()
                loop()
                eval_inference_end = time.time()

                # MLflow does not have the functionality to log artifacts per training step, so we have to incorporate
                # the training step (iteration_idx) into the artifact path
                eval_mlflow_start = time.time()
                if self.do_visualize:
                    mlflow.log_artifacts(temp_dir, 'iteration_%07i' % self.iteration_idx)
                eval_mlflow_end = time.time()

                if self.do_visualize:
                    shutil.rmtree(temp_dir)

                eval_end = time.time()

                if MLFLOW_PROFILING:
                    print(f'\nEvaluation took {"%.4f" % (eval_end - eval_start)}s in total; '
                          f'inference took {"%.4f" % (eval_inference_end - eval_inference_start)}s; '
                          f'MLflow logging took {"%.4f" % (eval_mlflow_end - eval_mlflow_start)}s '
                          f'(processed {sample_idx} sample(s))')
            self.iteration_idx += 1

    def _fit_model(self, mlflow_run):
        train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                               preprocessing=self.preprocessing)
        test_loader = self.dataloader.get_testing_dataloader(split=self.split, batch_size=1,
                                                             preprocessing=self.preprocessing)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = self.model.to(device)
        callback_handler = TorchTrainer.Callback(self, mlflow_run, model)
        for epoch in range(self.num_epochs):
            self._train_step(model, device, train_loader, callback_handler=callback_handler)
            # self._eval_step(model, device, test_loader)

    def _train_step(self, model, device, train_loader, callback_handler):
        model.train()
        opt = self.optimizer_or_lr
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = self.loss_function(model(x), y)
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss += self.loss_function(model(x), y, reduction='sum').item()
        test_loss /= len(test_loader.dataset)
        print(f'\nloss: {test_loss:.3f}')

    def _train(self):
        if self.mlflow_experiment_name is not None:
            self._init_mlflow()
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_experiment_name) as run:
                self._fit_model(mlflow_run=run)
        else:
            self._fit_model(mlflow_run=None)
