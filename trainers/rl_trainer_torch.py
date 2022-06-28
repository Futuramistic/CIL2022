import abc
import datetime
import functools
import itertools
import math
import numpy as np
import pysftp
import requests
from requests.auth import HTTPBasicAuth
from sklearn.utils import shuffle
import torch
import torch.cuda
from torch.utils.data import DataLoader, Subset
from urllib.parse import urlparse
from collections import namedtuple, deque

from losses.precision_recall_f1 import *
from models.Reinforcement.first_try import ReplayMemory
from models.Reinforcement.environment import SegmentationEnvironment, TERMINATE_NO, TERMINATE_YES
from utils.logging import mlflow_logger
from .trainer_torch import TorchTrainer
from utils import *

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import torch.distributions


# TODO: try beta distribution for policy network's outputs!


class TorchRLTrainer(TorchTrainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, patch_size=[100,100], history_size=5, max_rollout_len=1e6, std=1e-3):
        
        """
        Trainer for RL-based models.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the policy network to train
            ...
            patch_size: (int, int) the size of the observations for the actor
            history_size: (int) how many steps the actor can look back (become part of the observation)
            max_rollout_len: (int) how large each rollout can get on maximum
            std: (float) standard deviation assumed on the prediction of the actor network to use on a beta distribution as a policy to compute the next action
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, loss_function_hyperparams, evaluation_interval, num_samples_to_visualize,
                         checkpoint_interval, load_checkpoint_path, segmentation_threshold)
        self.patch_size = patch_size
        self.history_size = history_size
        self.max_rollout_len = max_rollout_len
        self.std=std
        # self.preprocessing and self.scheduler set by TorchTrainer superclass

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
            self.epoch_idx = 0
            self.epoch_iteration_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0
            self.f1_score = None

        def on_train_batch_end(self):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                precision, recall, self.f1_score = self.trainer.get_precision_recall_F1_score_validation()
                metrics = {'precision': precision, 'recall': recall, 'f1_score': self.f1_score}
                print('Metrics at aggregate iteration %i (ep. %i, ep.-it. %i): %s'
                      % (self.iteration_idx, self.epoch_idx, self.epoch_iteration_idx, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=self.iteration_idx)
                    if self.do_visualize:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
                
                if self.trainer.do_checkpoint\
                   and self.iteration_idx % self.trainer.checkpoint_interval == 0\
                   and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                    self.trainer._save_checkpoint(self.trainer.model, self.epoch_idx, self.epoch_iteration_idx, self.iteration_idx)

            self.iteration_idx += 1
            self.epoch_iteration_idx += 1
    
        def on_epoch_end(self):
            self.epoch_idx += 1
            self.epoch_iteration_idx = 0
            return self.f1_score
    
    class ReplayMemory(object):
        # inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html
        def __init__(self, capacity,):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(tuple(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path):
        return super().create_visualizations(file_path)

    def _save_checkpoint(self, model, epoch, epoch_iteration, total_iteration):
        return super()._save_checkpoint(model, epoch, epoch_iteration, total_iteration)

    def _load_checkpoint(self, checkpoint_path):
        return super()._load_checkpoint(checkpoint_path)

    def _fit_model(self, mlflow_run):
        return super()._fit_model(mlflow_run)

    def _train_step(self, model, device, train_loader, callback_handler):
        # WARNING: some models subclassing TorchTrainer overwrite this function, so make sure any changes here are
        # reflected appropriately in these models' files
        
        # TODO: parallelize with SubprocVecEnv
    
        def get_beta_params(mu, sigma):
            # returns (alpha, beta) parameters for Beta distribution, based on mu and sigma for that distribution
            # based on https://stats.stackexchange.com/a/12239
            alpha = ((1 - mu) / (sigma ** 2) - 1/mu) * mu ** 2
            beta = alpha * (1 / mu - 1)
            return alpha, beta

        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0

        for (x, y) in train_loader:
            for sample_x, sample_y in x, y:
                sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
                # insert each patch into ReplayMemory
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.patch_size, self.history_size, train_loader.img_val_min, train_loader.img_val_max) # take standard reward
                memory = ReplayMemory(capacity = self.max_rollout_len)
                env.reset()
                # "state" is input to model
                # observation in init state: RGB (3), history (5 by default), brush state (1)
                # use neutral action that does not affect anything to get initial state
                # env.step returns: (new_observation, reward, done, new_info)
                # "state" in tutorial is "observation" here
                observation, _, _, _ = env.step(env.get_neutral_action())
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate

                eps_reward = 0.0

                for timestep_idx in range(self.max_rollout_len):  # loop until model terminates or max len reached
                    # env.step(): action
                    model_output = self.model(observation)
                    
                    # calculate probs
                    # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'

                    # policy network outputs parameters of action distributions, then we sample from distributions
                    # parameterized by outputs of network
                    # choose sharp peak for Beta distributions to reduce stochasticity!
                    # assume Beta distribution for delta_angle (unscaled, in [0, 1])
                    # -> normalize to ((delta_angle + 1) / 2) before inputting to probability density function; after sampling, unnormalize output by -1 + 2 * sample(Beta( get_beta_params(center=((delta_angle + 1) / 2)) ))
                    # assume Beta distribution for magnitude, scaled to [0, patch_size / 2]
                    # assume Beta distribution for brush_state (unscaled, in [0, 1]) (here, peak should be especially sharp!)
                    # assume Beta distribution for brush_radius [0, patch_size / 2]
                    # assume Beta distribution for terminate (unscaled, in [0, 1])

                    action_log_probabilities = []
                    sampled_actions = []

                    # we assume all outputs are in [0, 1]
                    # we first use the same variance for all distributions
                    for idx, action_name in enumerate(['delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate']):
                        alpha, beta = get_beta_params(mu=model_output[:, idx], sigma=self.std)
                        dist = torch.distributions.Beta(alpha, beta)
                        sampled_action = dist.sample(1)
                        sampled_actions.append(sampled_action)
                        action_log_probabilities.append(dist.log_prob(sampled_action))

                        delta_angle = -1 + 2 * action[0]  # in [-1, 1] (later changed to [-pi, pi] in SegmentationEnvironment)
                        self.magnitude = action[1] * (self.min_patch_size / 2)  # 
                        new_brush_state = torch.round(action[2])  # tanh --> [-1, 1] 
                        # sigmoid stretched --> float [0, min_patch_size]
                        new_brush_radius = action[3] * (self.min_patch_size / 2)
                        # sigmoid rounded --> float [0, 1]
                        self.terminated = torch.round(action[4])   
                    
                    # TODO: test using different sigmas
                    # TODO: adapt Environment to output only action values in [0, 1] 
                    # TODO: optimize model
                    # TODO: MLFlow
                    # TODO: Visualization in real time and afterwards as animation, log viszualization to mlflow
                    
                    new_observation, reward, terminated, info = env.step(sampled_actions)
                
                        
                    memory.push(observation, model_output, new_observation, action_log_probabilities, sampled_actions, reward)
                    
                    observation = new_observation
                    eps_reward += reward
                    if terminated:
                        break
                    

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
            callback_handler.on_epoch_end()
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
        return super().get_F1_score_validation()

    def get_precision_recall_F1_score_validation(self):
        return super().get_precision_recall_F1_score_validation()
