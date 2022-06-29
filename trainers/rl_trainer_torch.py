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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from urllib.parse import urlparse
from collections import namedtuple, deque

from losses.precision_recall_f1 import *
from models.reinforcement.environment import SegmentationEnvironment, TERMINATE_NO, TERMINATE_YES
from utils.logging import mlflow_logger
from .trainer_torch import TorchTrainer
from utils import *

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import torch.distributions


# TODO: try letting the policy network output sigma parameters of distributions as well!

class TorchRLTrainer(TorchTrainer):
    def __init__(self, dataloader, model, preprocessing=None,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, history_size=5,
                 max_rollout_len=int(1e6), std=1e-3, reward_discount_factor=0.99, num_policy_epochs=4, policy_batch_size=10,
                 sample_from_action_distributions=False):
        
        """
        Trainer for RL-based models.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the policy network to train
            ...
            patch_size (int, int): the size of the observations for the actor
            history_size (int): how many steps the actor can look back (become part of the observation)
            max_rollout_len (int): how large each rollout can get on maximum
            std (float): standard deviation assumed on the prediction of the actor network to use on a beta distribution as a policy to compute the next action
            reward_discount_factor (float): factor by which to discount the reward per timestep into the future, when calculating the accumulated future return
            num_policy_epochs (int): number of epochs for which to train the policy network during a single iteration (for a single sample)
            policy_batch_size (int): size of batches to sample from the replay memory of the policy per policy training epoch
            sample_from_action_distributions (bool): controls whether to sample from the action distributions or to always use the mean
        """
        if loss_function is not None:
            raise RuntimeError('Custom losses not supported by TorchRLTrainer')
        
        if preprocessing is None:
            preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        super().__init__(dataloader=dataloader, model=model, preprocessing=preprocessing,
                 experiment_name=experiment_name, run_name=run_name, split=split, num_epochs=num_epochs, batch_size=batch_size,
                 optimizer_or_lr=optimizer_or_lr, scheduler=scheduler, loss_function=loss_function, loss_function_hyperparams=loss_function_hyperparams,
                 evaluation_interval=evaluation_interval, num_samples_to_visualize=num_samples_to_visualize, checkpoint_interval=checkpoint_interval,
                 load_checkpoint_path=load_checkpoint_path, segmentation_threshold=segmentation_threshold)
        self.history_size = int(history_size)
        self.max_rollout_len = int(max_rollout_len)
        self.std = torch.tensor(std, device=self.device)
        self.reward_discount_factor = float(reward_discount_factor)
        self.num_policy_epochs = int(num_policy_epochs)
        self.sample_from_action_distributions = bool(sample_from_action_distributions)
        self.policy_batch_size = int(policy_batch_size)

        # self.scheduler set by TorchTrainer superclass

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
        # inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # and https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(tuple(args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def compute_accumulated_discounted_returns(self, gamma):
            # memory.push(observation, new_observation, model_output, action_log_probabilities, sampled_actions,
            #                   self.terminated, reward, torch.tensor(float('nan')))
            for step_idx in reversed(range(len(self.memory))):  # differs from tutorial code (but appears to be equivalent)
                terminated = self.memory[step_idx + 1][5]
                future_return = self.memory[step_idx + 1][-1] if step_idx < len(self.memory) else 0
                current_reward = self.memory[step_idx + 1][-1]
                self.memory[step_idx][-1] = future_return * gamma * (1 - terminated) + current_reward

        def __len__(self):
            return len(self.memory)

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path):
        num_to_visualize = self.num_samples_to_visualize
        num_fixed_samples = num_to_visualize // 2
        num_random_samples = num_to_visualize - num_fixed_samples
        # start sampling random indices from "num_fixed_samples + 1" to avoid duplicates
        # convert to np.array to allow subsequent slicing
        if num_to_visualize >= len(self.test_loader):
            # just visualize the entire test set
            indices = np.array(list(range(len(self.test_loader))))
        else:
            indices = np.array(list(range(num_fixed_samples)) +\
                            random.sample(range(num_fixed_samples + 1, len(self.test_loader)), num_random_samples))
        images = []
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)
        subset_ds = Subset(self.test_loader.dataset, indices)
        subset_dl = DataLoader(subset_ds, batch_size=vis_batch_size, shuffle=False)
        
        for (batch_xs, batch_ys) in subset_dl:
            # batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.numpy()
            # output = self.model(batch_xs)
            # if type(output) is tuple:
            #     output = output[0]
            # preds = (output >= self.segmentation_threshold).float().cpu().detach().numpy()

            for sample_x, sample_y in batch_xs, batch_ys:
                sample_x, sample_y =\
                    sample_x.to(self.device, dtype=torch.float32), sample_y.to(self.device, dtype=torch.long)
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size,
                                              self.test_loader.img_val_min, self.test_loader.img_val_max)
                env.reset()
                observation, _, _, _ = env.step(env.get_neutral_action())
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
                # loop until termination/timeout already inside this function
                self.trajectory_step(self, env, observation,
                                     sample_from_action_distributions=self.sample_from_action_distributions)
                preds = env.get_unpadded_segmentation().float()

                # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
                self._fill_images_array(preds.unsqueeze(0), sample_y.unsqueeze(0), images)

        self._save_image_array(images, file_path)

        # return super().create_visualizations(file_path)

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

        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0

        for (x, y) in train_loader:
            for idx, sample_x in enumerate(x):
                sample_y = y[idx]
                sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
                # insert each patch into ReplayMemory
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, train_loader.img_val_min, train_loader.img_val_max) # take standard reward
                memory = self.ReplayMemory(capacity = int(self.max_rollout_len))
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

                # if we exit this function and "terminated" is false, we timed out
                # (more than self.max_rollout_len iterations)
                # loop until termination/timeout already inside this function
                self.trajectory_step(env, observation,
                                    sample_from_action_distributions=self.sample_from_action_distributions,
                                    memory=memory)

                memory.compute_accumulated_discounted_returns(gamma=self.reward_discount_factor)

                for epoch_idx in range(self.num_policy_epochs):
                    policy_batch = memory.sample(self.policy_batch_size)
                    for sample in policy_batch:
                        # memory.push(observation, new_observation, model_output, action_log_probabilities, sampled_actions,
                        #                   self.terminated, reward, torch.tensor(float('nan')))
                        observation_batch, new_observation_batch, model_output_batch, action_log_probabilities_batch,\
                            sampled_actions_batch, terminated_batch, reward_batch, returns_batch = sample

                        policy_loss = -(action_log_probabilities_batch * returns_batch).mean()
                        # TODO: add entropy loss and see if it helps
                        # entropy_loss = ...
                        loss = policy_loss
                        opt.zero_grad()
                        loss.backward(retain_graph=False)
                        opt.step()
                        with torch.no_grad():
                            train_loss += loss.item()
                memory.reset()
                
            train_loss /= (len(train_loader.dataset) * self.policy_batch_size * self.num_policy_epochs)
            callback_handler.on_epoch_end()
            self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0

        for (x, y) in test_loader:
            for sample_x, sample_y in x, y:
                sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, test_loader.img_val_min, test_loader.img_val_max) # take standard reward
                env.reset()
                # "state" is input to model
                # observation in init state: RGB (3), history (5 by default), brush state (1)
                # use neutral action that does not affect anything to get initial state
                # env.step returns: (new_observation, reward, done, new_info)
                # "state" in tutorial is "observation" here
                observation, _, _, _ = env.step(env.get_neutral_action())
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
                # loop until termination/timeout already inside this function
                self.trajectory_step(self, env, observation, sample_from_action_distributions=False)
                preds = env.get_unpadded_segmentation().float()
                test_loss += self.loss_function(preds, sample_y).item()

        test_loss /= len(test_loader.dataset)
        return test_loss
    
    def trajectory_step(self, env, observation, sample_from_action_distributions=None, memory=None):
        """Uses model predictions to create trajectory until terminated

        Args:
            env (Environment): the environment the model shall explore
            observation (Observation): the observation input for the model
            sample_from_action_distributions (bool, optional): Whether to sample from the action distribution created by the model output or use model output directly. Defaults to self.sample_action_distribution.
            memory (Memory, optional): Memory to store the trajectory during training. Defaults to None.
        """
        def get_beta_params(mu, sigma):
            # returns (alpha, beta) parameters for Beta distribution, based on mu and sigma for that distribution
            # based on https://stats.stackexchange.com/a/12239
            alpha = ((1 - mu) / (sigma ** 2) - 1/mu) * mu ** 2
            beta = alpha * (1 / mu - 1)
            return alpha, beta
        
        eps_reward = 0.0

        if sample_from_action_distributions is None:
            sample_from_action_distributions = self.sample_from_action_distributions

        for timestep_idx in range(self.max_rollout_len):  # loop until model terminates or max len reached
            model_output = self.model(observation.unsqueeze(0))

            # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate', all in [0,1]
            # we use the output of the network as the direct prediction instead of sampling like during training
            # we assume all outputs are in [0, 1]
            
            # we assume all outputs are in [0, 1]
            # we first use the same variance for all distributions
            
            # action: ['delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate']
            # define 5 distributions at once and sample from them
            action_log_probabilities = None
            if sample_from_action_distributions:
                action_mus = model_output[0]  # remove batch dimension
                alphas, betas = get_beta_params(mu=action_mus, sigma=self.std)
                dist = torch.distributions.Beta(alphas, betas)
                action = dist.sample()
                action_log_probabilities = dist.log_prob(action)
            else:
                action = model_output[0]  # remove batch dimension

            action[0] = -1 + 2 * action[0]  # delta_angle in [-1, 1] (later changed to [-pi, pi] in SegmentationEnvironment)
            action[1] = action[1] * (env.min_patch_size / 2)  # magnitude
            action[2] = torch.round(-1 + 2 * action[2])  # new_brush_state
            # sigmoid stretched --> float [0, min_patch_size]
            action[3] = action[3] * (env.min_patch_size / 2) # new_brush_radius
            # sigmoid rounded --> float [0, 1]
            action[4] = torch.round(action[4]) # terminated 

            new_observation, reward, terminated, info = env.step(action)
            
            if memory is not None:
                memory.push(observation, new_observation, model_output, action_log_probabilities, action,
                                terminated, reward, torch.tensor(float('nan')))

            observation = new_observation
            eps_reward += reward
            if terminated >= 0.5:
                break
        
        return observation, reward, eps_reward, terminated, info, action_log_probabilities

    def get_F1_score_validation(self):
        return super().get_F1_score_validation()

    def get_precision_recall_F1_score_validation(self):
        self.model.eval()
        precisions, recalls, f1_scores = [], [], []
        for (x, y) in self.test_loader:
            for sample_x, sample_y in x, y:
                sample_x, sample_y = sample_x.to(self.device, dtype=torch.float32), sample_y.to(self.device, dtype=torch.long)
                
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, self.test_loader.img_val_min, self.test_loader.img_val_max) # take standard reward
                env.reset()

                observation, _, _, _ = env.step(env.get_neutral_action())

                # loop until termination/timeout already inside this function
                self.trajectory_step(self, env, observation, sample_from_action_distributions=False)
                
                preds = env.get_unpadded_segmentation().float()
                precision, recall, f1_score = precision_recall_f1_score_torch(preds, y)
                precisions.append(precision.cpu().numpy())
                recalls.append(recall.cpu().numpy())
                f1_scores.append(f1_score.cpu().numpy())

        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self, param)
                   for param in ['history_size', 'max_rollout_len', 'std', 'reward_discount_factor', 
                                 'num_policy_epochs', 'policy_batch_size', 'sample_from_action_distributions']
                   if hasattr(self, param)}),
                **({param: getattr(self.model, param)
                   for param in ['patch_size']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        return optim.Adam(model.parameters(), lr=1e-4)
