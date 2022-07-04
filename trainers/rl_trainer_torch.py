import abc
import datetime
import functools
import itertools
import math
import numpy as np
import pysftp
import random
import requests
from requests.auth import HTTPBasicAuth
from sklearn.utils import shuffle
import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from urllib.parse import urlparse
from collections import namedtuple

from losses.precision_recall_f1 import *
from models.reinforcement.environment import SegmentationEnvironment, TERMINATE_NO, TERMINATE_YES
from utils.logging import mlflow_logger
from .trainer_torch import TorchTrainer
from utils import *

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import torch.distributions

import imageio

from PIL import Image, ImageDraw


# TODO: try letting the policy network output sigma parameters of distributions as well!

EPSILON = 1e-5

class TorchRLTrainer(TorchTrainer):
    def __init__(self, dataloader, model, preprocessing=None,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, history_size=5,
                 max_rollout_len=int(2*16e4), replay_memory_capacity=int(1e4), std=[1e-3, 1e-3, 1e-3, 1e-3, 1e-2], reward_discount_factor=0.99,
                 num_policy_epochs=4, policy_batch_size=10, sample_from_action_distributions=False, visualization_interval=20,
                 min_steps=20):
        """
        Trainer for RL-based models.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the policy network to train
            ...
            patch_size (int, int): the size of the observations for the actor
            history_size (int): how many steps the actor can look back (become part of the observation)
            max_rollout_len (int): how large each rollout can get on maximum
                                   default value: 2 * 160000 (image size: 400*400; give agent chance to visit each pixel twice)
            replay_memory_capacity (int): capacity of the replay memory
            std ([float, float, float, float, float]): standard deviation assumed on the predictions 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate' of the actor network to use on a Normal distribution as a policy to compute the next action
            reward_discount_factor (float): factor by which to discount the reward per timestep into the future, when calculating the accumulated future return
            num_policy_epochs (int): number of epochs for which to train the policy network during a single iteration (for a single sample)
            policy_batch_size (int): size of batches to sample from the replay memory of the policy per policy training epoch
            sample_from_action_distributions (bool): controls whether to sample from the action distributions or to always use the mean
            visualization_interval (int): logs the predicted trajectory as a gif every <visualization_interval> steps.
            min_steps (int): minimum number of steps agent has to perform before it is allowed to terminate a trajectory
        """
        if loss_function is not None:
            raise RuntimeError('Custom losses not supported by TorchRLTrainer')
        
        if preprocessing is None:
            preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        if optimizer_or_lr is None:
            optimizer_or_lr = TorchRLTrainer.get_default_optimizer_with_lr(1e-4, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = TorchRLTrainer.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_or_lr, lr_lambda=lambda epoch: 1.0)
        
        if not isinstance(std, list):
            std = [std]*5

        super().__init__(dataloader=dataloader, model=model, preprocessing=preprocessing,
                 experiment_name=experiment_name, run_name=run_name, split=split, num_epochs=num_epochs, batch_size=batch_size,
                 optimizer_or_lr=optimizer_or_lr, scheduler=scheduler, loss_function=loss_function, loss_function_hyperparams=loss_function_hyperparams,
                 evaluation_interval=evaluation_interval, num_samples_to_visualize=num_samples_to_visualize, checkpoint_interval=checkpoint_interval,
                 load_checkpoint_path=load_checkpoint_path, segmentation_threshold=segmentation_threshold)
        self.history_size = int(history_size)
        self.max_rollout_len = int(max_rollout_len)
        self.replay_memory_capacity = int(replay_memory_capacity)
        self.std = torch.tensor(std, device=self.device).detach()
        self.reward_discount_factor = float(reward_discount_factor)
        self.num_policy_epochs = int(num_policy_epochs)
        self.sample_from_action_distributions = bool(sample_from_action_distributions)
        self.policy_batch_size = int(policy_batch_size)
        self.visualization_interval = int(visualization_interval)
        self.min_steps = int(min_steps)

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
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx, self.epoch_idx, self.epoch_iteration_idx)
                
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
        def __init__(self, capacity, balancing_trajectory_length=None):
            # balancing_trajectory_length: if not None, specifies whether to randomly discard or keep newly added elements when the memory
            # is full, in a way that ensures the buffer contains samples evenly spread across trajectories of the length "capacity"
            # better than using the maximum length of a trajectory would be to use the expected length of a trajectory
            self.capacity = capacity
            self.balancing_trajectory_length = balancing_trajectory_length
            self.memory = []

        def push(self, *args):
            """Save a transition"""
            new_val = [tensor.clone() for tensor in args]
            if len(self.memory) >= self.capacity:
                if self.balancing_trajectory_length is not None and random.uniform(0.0, 1.0) > self.capacity / self.balancing_trajectory_length:
                    # discard new sample to ensure even spread of samples across trajectory
                    return
                self.memory[random.randint(0, len(self.memory) - 1)] = new_val
            else:
                self.memory.append(new_val)

        def sample(self, batch_size):
            return random.sample(self.memory, min(batch_size, len(self.memory)))

        def compute_accumulated_discounted_returns(self, gamma):
            # memory.push(observation, model_output, sampled_actions,
            #                   self.terminated, reward, torch.tensor(float('nan')))
            for step_idx in reversed(range(len(self.memory))):  # differs from tutorial code (but appears to be equivalent)
                terminated = self.memory[step_idx][3] 
                future_return = self.memory[step_idx + 1][-1] if step_idx < len(self.memory) - 1 else 0
                current_reward = self.memory[step_idx][4]
                self.memory[step_idx][-1] = current_reward + future_return * gamma * (1 - terminated)

        def __len__(self):
            return len(self.memory)

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path, iteration_index, epoch_idx, epoch_iteration_idx):
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
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)
        subset_ds = Subset(self.test_loader.dataset, indices)
        subset_dl = DataLoader(subset_ds, batch_size=vis_batch_size, shuffle=False)
        
        predictions_nstepwise_rgb_test_set = []
        positions_nstepwise_test_set = []

        for (batch_xs, batch_ys) in subset_dl:
            # batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.numpy()
            # output = self.model(batch_xs)
            # if type(output) is tuple:
            #     output = output[0]
            # preds = (output >= self.segmentation_threshold).float().cpu().detach().numpy()

            preds_batch = []
            longest_animation_length = 0
            
            for idx, sample_x in enumerate(batch_xs):
                sample_y = batch_ys[idx]
                sample_x, sample_y_gpu =\
                    sample_x.to(self.device, dtype=torch.float32), sample_y.to(self.device, dtype=torch.long)
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y_gpu = torch.squeeze(sample_y_gpu)
                env = SegmentationEnvironment(sample_x, sample_y_gpu, self.model.patch_size, self.history_size,
                                              self.test_loader.img_val_min, self.test_loader.img_val_max)
                env.reset()
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
                # loop until termination/timeout already inside this function
                predictions_nstepwise, positions_nstepwise =\
                    self.trajectory_step(env, sample_from_action_distributions=self.sample_from_action_distributions, visualization_interval=self.visualization_interval)
                predictions_nstepwise_np = np.stack(predictions_nstepwise)  # stack all animation frames, dim(nsteps, height, width)
                
                sample_y_np = sample_y.numpy()
                
                merged = np.expand_dims(2 * predictions_nstepwise_np + sample_y_np, axis=0)
                green_channel = merged == 3  # true positives
                red_channel = merged == 2  # false positive
                blue_channel = merged == 1  # false negative
                rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=0) # shape(rgb, nsteps, height, width)

                predictions_nstepwise_rgb_test_set.append(rgb)
                positions_nstepwise_test_set.append(positions_nstepwise)
                
                if len(predictions_nstepwise) > longest_animation_length:
                    longest_animation_length = len(predictions_nstepwise)

                preds = env.get_unpadded_segmentation().float().detach().cpu()
                preds_batch.append(preds)

        # calculate grid size
        
        n = len(predictions_nstepwise_rgb_test_set)  # test_set_size
        if is_perfect_square(n):
            nb_cols = math.sqrt(n)
        else:
            nb_cols = math.sqrt(next_perfect_square(n))
        nb_cols = int(nb_cols)  # Need it to be an integer
        nb_rows = math.ceil(float(n) / float(nb_cols))  # Number of rows in final image
        
        # dimensions of predictions_nstepwise_rgb_test_set:
        # [test_set_size (list), RGB (np.ndarray), timesteps (np.ndarray), image height (np.ndarray), image width (np.ndarray)]

        # generate GIF
        if len(predictions_nstepwise_rgb_test_set) > 0:
            _, _, img_height, img_width = predictions_nstepwise_rgb_test_set[0].shape # shape(rgb, nsteps, height, width)
            
            gif_filename = f"globaliter_{iteration_index}_epoch_{epoch_idx}_epochiter_{epoch_iteration_idx}.gif"
            gif_path = os.path.join(os.path.dirname(file_path), gif_filename)

            # 1, 10, 10, 10

            gif_frames = []
            for timestep_idx in range(longest_animation_length):
                # fill with zeros to ensure background is black
                uber_img = np.zeros((3, nb_rows * img_height, nb_cols * img_width), dtype=np.uint8) # shape(3, 3*img_height, 3*img_width)
                for prediction_idx, prediction in enumerate(predictions_nstepwise_rgb_test_set):
                    # prediction has shape(rgb, nsteps, height, width)
                    prediction_len = prediction.shape[1]  # time dimension

                    row_idx = prediction_idx // nb_cols
                    col_idx = int(prediction_idx % nb_cols)
                    
                    start_y = row_idx * img_height
                    end_y = start_y + img_height
                    start_x = col_idx * img_width
                    end_x = start_x + img_width

                    # sample's frame counts (prediction_len of each sample): [2, 10]
                    # timestep_idx = 0: [0, 0]  << if prediction_len > timestep_idx
                    # timestep_idx = 1: [1, 1]  << if prediction_len > timestep_idx
                    # timestep_idx = 2: [1, 2]  << else
                    # timestep_idx = 3: [1, 3]  << else
                    # ...

                    # prediction[:, timestep_idx]*255): shape(rgb, height, width)


                    agent_pos = positions_nstepwise_test_set[prediction_idx][min(timestep_idx, prediction_len-1)] # x, y
                    agent_pos_patch_size = 3
                    agent_pos_patch_start_x = max(0, agent_pos[0] - agent_pos_patch_size)
                    agent_pos_patch_start_y = max(0, agent_pos[1] - agent_pos_patch_size)
                    agent_pos_patch_end_x = min(img_width - 1, agent_pos[0] + agent_pos_patch_size)
                    agent_pos_patch_end_y = min(img_height - 1, agent_pos[1] + agent_pos_patch_size)
                    
                    curr_pred = prediction[:, min(timestep_idx, prediction_len-1)]  # collapse time dimension
                    curr_pred[:, agent_pos_patch_start_x:agent_pos_patch_end_x, agent_pos_patch_start_y:agent_pos_patch_end_y] =\
                        np.array([[[255]], [[255]], [[51]]], dtype=np.uint8) # signalize agent position

                    # take either the prediction for the current timestep or the last prediction (the latter if the current timestep exceeds the sample length)
                    uber_img[:, start_y:end_y, start_x:end_x] = (curr_pred*255).astype(np.uint8)
                    if timestep_idx >= prediction_len - 5:  # signalize that current sample has no more frames
                        uber_img[:, (end_y - 2 * agent_pos_patch_size):end_y, (end_x - 2 * agent_pos_patch_size):end_x] = np.array([[[255]], [[0]], [[0]]], dtype=np.uint8)
                        
                    if timestep_idx >= longest_animation_length - 5: # signalize end of gif with a red mark on the bottom right
                        uber_img[:, -agent_pos_patch_size*4:, -agent_pos_patch_size*4:] = np.array([[[255]], [[0]], [[0]]], dtype=np.uint8)

                    
                gif_frame = Image.fromarray(uber_img.transpose((1, 2, 0)))
                draw = ImageDraw.Draw(gif_frame)
                draw.text((10, 10), str(timestep_idx * self.visualization_interval), fill=(137, 207, 240))  # baby blue

                gif_frames.append(gif_frame) # height, width, rgb
                        
            
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=100, loop=0) # duration is milliseconds between frames, loop=0 means loop infinite times
            # if not working, convert via: imageio.core.util.Array(numpy_array)

            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            # self._fill_images_array(torch.stack(preds_batch, dim=0).unsqueeze(1).float().detach().cpu().numpy(), batch_ys.numpy(), images)

        # return super().create_visualizations(file_path)
        return gif_path

    def _train_step(self, model, device, train_loader, callback_handler):
        # WARNING: some models subclassing TorchTrainer overwrite this function, so make sure any changes here are
        # reflected appropriately in these models' files
        
        # TODO: parallelize with SubprocVecEnv

        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0

        for (xs, ys) in train_loader:
            for idx, sample_x in enumerate(xs):
                sample_y = ys[idx]
                sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
                # insert each patch into ReplayMemory
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, train_loader.img_val_min, train_loader.img_val_max) # take standard reward
                # using maximum memory capacity for balancing_trajectory_length is suboptimal;
                # could e.g. empirically approximate expected trajectory length and use that instead
                memory = self.ReplayMemory(capacity=int(self.replay_memory_capacity),
                                           balancing_trajectory_length=int(self.replay_memory_capacity))
                
                # "state" is input to model
                # observation in init state: RGB (3), history (5 by default), brush state (1)
                
                # env.step returns: (new_observation, reward, done, new_info)
                # "state" in tutorial is "observation" here
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate

                eps_reward = 0.0

                # if we exit this function and "terminated" is false, we timed out
                # (more than self.max_rollout_len iterations)
                # loop until termination/timeout already inside this function
                self.trajectory_step(env,
                                    sample_from_action_distributions=self.sample_from_action_distributions,
                                    memory=memory)

                memory.compute_accumulated_discounted_returns(gamma=self.reward_discount_factor)

                for epoch_idx in range(self.num_policy_epochs):
                    loss_accumulator = []
                    policy_batch = memory.sample(self.policy_batch_size)
                    for sample_idx, sample in enumerate(policy_batch):
                        # memory.push(observation, model_output, action_log_probabilities, sampled_actions,
                        #                   self.terminated, reward, torch.tensor(float('nan')))

                        # model_output_sample, action_log_probabilities_sample, sampled_actions_sample: have requires_grad=True
                        # model_output_sample, sampled_actions_sample not even used
                        observation_sample, model_output_sample,\
                            sampled_actions_sample, terminated_sample, reward_sample, returns_sample = sample

                        # recalculate log probabilities (see tutorial: also recalculated)
                        # if we do not recalculate, we will get 

                        model_output = self.model(observation_sample.unsqueeze(0))

                        # mean = alpha / (alpha + beta)

                        # alphas, betas = TorchRLTrainer.get_beta_params(mu=model_output[0], sigma=self.std)  # remove batch dimension, multiply and add to avoid extreme points
                        # print(f'alphas, betas: {(alphas, betas)}')
                        
                        # dist = torch.distributions.Beta(alphas, betas)
                        covariance_matrix = torch.diag(self.std)
                        dist = torch.distributions.MultivariateNormal(loc=model_output[0], covariance_matrix=covariance_matrix)
                        action_log_probabilities_sample = dist.log_prob(model_output_sample)

                        policy_loss = -(action_log_probabilities_sample * returns_sample).mean()
                        # TODO: add entropy loss and see if it helps
                        # entropy_loss = ...
                            
                        opt.zero_grad()
                        loss = policy_loss
                        loss.backward(retain_graph=True) # retain computational graph because needed when sampling the same trajectory multiple time
                        for param in self.model.parameters(): # gradient clipping
                            param.grad.data.clamp_(-1, 1)
                        opt.step()
                        
                        with torch.no_grad():
                            train_loss += policy_loss.item()
                
            callback_handler.on_train_batch_end()
            
            train_loss /= (len(train_loader.dataset) * self.policy_batch_size * self.num_policy_epochs)
            callback_handler.on_epoch_end()
            self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        if self.loss_function is None:
            return 0.0

        model.eval()
        test_loss = 0

        for (xs, ys) in test_loader:
            for idx, sample_x in enumerate(xs):
                sample_y = ys[idx]
                sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, test_loader.img_val_min, test_loader.img_val_max) # take standard reward

                # "state" is input to model
                # observation in init state: RGB (3), history (5 by default), brush state (1)
                # env.step returns: (new_observation, reward, done, new_info)
                # "state" in tutorial is "observation" here
                
                # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
                # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
                # loop until termination/timeout already inside this function
                self.trajectory_step(env, sample_from_action_distributions=False)
                preds = env.get_unpadded_segmentation().float()
                test_loss += self.loss_function(preds, sample_y).item()

        test_loss /= len(test_loader.dataset)
        return test_loss

    @staticmethod
    def get_beta_params(mu, sigma):
        # must ensure that sigma**2.0 <= mu * (1 - mu)
        if not hasattr(sigma, 'shape') and hasattr(mu, 'shape'):  # mu is tensor, sigma is scalar
            sigma = sigma * torch.ones_like(mu)
        elif not hasattr(mu, 'shape'):  # mu and sigma are scalars
            mu = torch.tensor(mu)
            sigma = torch.tensor(sigma)
        
        # mu = torch.clamp(mu, EPSILON/2, 1-EPSILON/2)

        mu = torch.clamp(mu, EPSILON/2, 1 - EPSILON/2)
        sigma = torch.clamp(sigma, torch.zeros_like(sigma) + 1e-8, torch.sqrt(mu * (1.0 - mu)))
        
        # returns (alpha, beta) parameters for Beta distribution, based on mu and sigma for that distribution
        # based on https://stats.stackexchange.com/a/12239
        alpha = ((1 - mu) / (sigma ** 2) - 1/mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta

    def trajectory_step(self, env, sample_from_action_distributions=None, memory=None, visualization_interval=-1):
        """Uses model predictions to create trajectory until terminated
        Args:
            env (Environment): the environment the model shall explore
            observation (Observation): the observation input for the model
            sample_from_action_distributions (bool, optional): Whether to sample from the action distribution created by the model output or use model output directly. Defaults to self.sample_action_distribution.
            memory (Memory, optional): Memory to store the trajectory during training. Defaults to None.
            visualization_interval (int, optional): number of timesteps between frames of the returned list of observations
        Returns:
            list of observations that can be used to generate animations of the agent's trajectory, or [] if visualization_interval <= 0 (List of torch.Tensor)
        """

        env.reset()
        predictions_nsteps = [env.get_unpadded_segmentation().float().detach().cpu().numpy()]
        positions_nsteps = [env.agent_pos]

        observation, _, _, _ = env.step(env.get_neutral_action())    
        predictions_nsteps.append(env.get_unpadded_segmentation().float().detach().cpu().numpy())
        positions_nsteps.append(env.agent_pos)

        eps_reward = 0.0

        if sample_from_action_distributions is None:
            sample_from_action_distributions = self.sample_from_action_distributions

        for timestep_idx in range(self.max_rollout_len):  # loop until model terminates or max len reached
            model_output = self.model(observation.detach().unsqueeze(0)).detach()

            # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate', all in [0,1]
            # we use the output of the network as the direct prediction instead of sampling like during training
            # we assume all outputs are in [0, 1]
            
            # we assume all outputs are in [0, 1]
            # we first use the same variance for all distributions
            
            # action: ['delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate']
            # define 5 distributions at once and sample from them
    
            # alphas, betas = TorchRLTrainer.get_beta_params(mu=model_output[0], sigma=self.std)  # remove batch dimension
            # dist = torch.distributions.Beta(alphas, betas)
            
            if sample_from_action_distributions:
                action = torch.normal(mean=model_output, std=self.std)[0]
                action = torch.clamp(action, EPSILON/2, 1-EPSILON/2)
            else:
                action = model_output[0]  # remove batch dimension

            action_rounded_list = []
            action_rounded_list.append(-1 + 2 * action[0])  # delta_angle in [-1, 1] (later changed to [-pi, pi] in SegmentationEnvironment)
            action_rounded_list.append(action[1] * (env.min_patch_size / 2))  # magnitude
            action_rounded_list.append(torch.round(-1 + 2 * action[2]))  # new_brush_state
            # sigmoid stretched --> float [0, min_patch_size]
            action_rounded_list.append(action[3] * (env.min_patch_size / 2)) # new_brush_radius
            # sigmoid rounded --> float [0, 1]
            action_rounded_list.append(torch.gt(action[4], 0.9).float()) # terminated only if output bigger than 0.9

            action_rounded = torch.cat([tensor.unsqueeze(0) for tensor in action_rounded_list], dim=0)

            new_observation, reward, terminated, info = env.step(action)
            
            if memory is not None:
                memory.push(observation, model_output, action_rounded, terminated, reward, torch.tensor(float('nan')))

            observation = new_observation
            eps_reward += reward
            
            if visualization_interval > 0 and timestep_idx % visualization_interval == 0:
                predictions_nsteps.append((env.get_unpadded_segmentation().float().detach().cpu().numpy()).astype(int))
                positions_nsteps.append(env.agent_pos)
            
            if terminated and timestep_idx >= self.min_steps:
                break
        
        return predictions_nsteps, positions_nsteps

    def get_precision_recall_F1_score_validation(self):
        self.model.eval()
        precisions, recalls, f1_scores = [], [], []
        for (xs, ys) in self.test_loader:
            for idx, sample_x in enumerate(xs):
                sample_y = ys[idx]
                sample_x, sample_y = sample_x.to(self.device, dtype=torch.float32), sample_y.to(self.device, dtype=torch.long)
                
                # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
                # accordingly, sample_y must be of shape (H, W) not (1, H, W)
                sample_y = torch.squeeze(sample_y)
                env = SegmentationEnvironment(sample_x, sample_y, self.model.patch_size, self.history_size, self.test_loader.img_val_min, self.test_loader.img_val_max) # take standard reward

                # loop until termination/timeout already inside this function
                self.trajectory_step(env, sample_from_action_distributions=False)
                
                preds = env.get_unpadded_segmentation().float()
                precision, recall, f1_score = precision_recall_f1_score_torch(preds, sample_y)
                precisions.append(precision.cpu().numpy())
                recalls.append(recall.cpu().numpy())
                f1_scores.append(f1_score.cpu().numpy())

        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self, param)
                   for param in ['history_size', 'max_rollout_len', 'reward_discount_factor',
                                 'num_policy_epochs', 'policy_batch_size', 'sample_from_action_distributions',
                                 'visualization_interval', 'min_steps']
                   if hasattr(self, param)}),
                **({k:v.item() for k,v in zip(['std_delta_angle', 'std_magnitude', 'std_brush_state', 'std_brush_radius', 'std_terminate'], self.std)}),
                **({param: getattr(self.model, param)
                   for param in ['patch_size']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        return optim.Adam(model.parameters(), lr=lr)
