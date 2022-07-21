# TODO: add mini-version of current segmentation map as input, so network knows history!

import math
import numpy as np
from datetime import datetime
import random
import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from losses.precision_recall_f1 import *
from models.reinforcement.environment import DEFAULT_REWARDS_MINIMAL, SegmentationEnvironmentMinimal
from utils.logging import mlflow_logger
from .trainer_torch import TorchTrainer
from utils import *

import gym
from subproc_vec_env_torch import SubprocVecEnvTorch

import torch.distributions

import imageio

from PIL import Image, ImageDraw


# TODO: try letting the policy network output sigma parameters of distributions as well!

EPSILON = 1e-5
log_debug_indent = 0


def log_debug(line, no_line_break=False, log_timestamp=True):
    global log_debug_indent
    to_print = ('[' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] ' + log_debug_indent * '  ' + line) if log_timestamp else line
    if no_line_break:
        print(to_print, end='', flush=True)
    else:
        print( to_print)


class TorchRLTrainerMinimal(TorchTrainer):
    def __init__(self, dataloader, model, preprocessing=None,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None, use_channelwise_norm=False,
                 rollout_len=int(2*16e4), replay_memory_capacity=int(1e4), std=[1e-3, 1e-3], reward_discount_factor=0.99,
                 num_policy_epochs=4, policy_batch_size=10, sample_from_action_distributions=False, visualization_interval=20,
                 rewards = None, blobs_removal_threshold=0, hyper_seg_threshold=False, use_supervision=False,
                 gradient_clipping_norm=1.0):
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
            replay_memory_capacity (int): capacity of the replay memory, per sample of the train loader!
            std ([float, float]): standard deviation assumed on the predictions 'delta_angle', 'brush_state' of the actor network to use on a Normal distribution as a policy to compute the next action
            reward_discount_factor (float): factor by which to discount the reward per timestep into the future, when calculating the accumulated future return
            num_policy_epochs (int): number of epochs for which to train the policy network during a single iteration (for a single sample)
            policy_batch_size (int): size of batches to sample from the replay memory of the policy per policy training epoch
            sample_from_action_distributions (bool): controls whether to sample from the action distributions or to always use the mean
            visualization_interval (int): logs the predicted trajectory as a gif every <visualization_interval> steps.
            min_steps (int): minimum number of steps agent has to perform before it is allowed to terminate a trajectory
            rewards (dict): Dictionary of rewards, see type of rewards in models/reinforcement/environment.py under DEFAULT_REWARDS
            use_supervision (bool): set to True to train the policy network using supervision by an automatically generated example trajectory for each sample, rather than
                                    training with rewards as in classical reinforcement learning
            gradient_clipping_norm (float): norm to clip the gradients of the loss function to; set to 0 or None to avoid gradient clipping
        """
        if loss_function is not None and not use_supervision:
            raise RuntimeError('Custom losses not supported by TorchRLTrainer in non-supervised mode')
        
        if preprocessing is None:
            if use_channelwise_norm and dataloader.dataset in DATASET_STATS:
                def channelwise_preprocessing(x, is_gt):
                    if is_gt:
                        return x[:1, :, :].float() / 255
                    stats = DATASET_STATS[dataloader.dataset]
                    x = x[:3, :, :].float()
                    x[0] = (x[0] - stats['pixel_mean_0']) / stats['pixel_std_0']
                    x[1] = (x[1] - stats['pixel_mean_1']) / stats['pixel_std_1']
                    x[2] = (x[2] - stats['pixel_mean_2']) / stats['pixel_std_2']
                    return x
                preprocessing = channelwise_preprocessing
            else:
                # convert samples to float32 \in [0, 1] & remove A channel;
                # convert ground truth to int \in {0, 1} & remove A channel
                preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)

        if optimizer_or_lr is None:
            optimizer_or_lr = TorchRLTrainerMinimal.get_default_optimizer_with_lr(1e-4, model)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = TorchRLTrainerMinimal.get_default_optimizer_with_lr(optimizer_or_lr, model)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_or_lr, lr_lambda=lambda epoch: 1.0)
            
        if loss_function is None:
            loss_function = torch.nn.MSELoss()
        
        if isinstance(std, int) or isinstance(std, float):
            std = [std]*2

        super().__init__(dataloader=dataloader, model=model, preprocessing=preprocessing,
                 experiment_name=experiment_name, run_name=run_name, split=split, num_epochs=num_epochs, batch_size=batch_size,
                 optimizer_or_lr=optimizer_or_lr, scheduler=scheduler, loss_function=loss_function, loss_function_hyperparams=loss_function_hyperparams,
                 evaluation_interval=evaluation_interval, num_samples_to_visualize=num_samples_to_visualize, checkpoint_interval=checkpoint_interval,
                 load_checkpoint_path=load_checkpoint_path, segmentation_threshold=segmentation_threshold, use_channelwise_norm=use_channelwise_norm,
                 blobs_removal_threshold=blobs_removal_threshold, hyper_seg_threshold=hyper_seg_threshold)
        self.rollout_len = int(rollout_len)
        self.replay_memory_capacity = int(replay_memory_capacity)
        self.std = torch.tensor(std, device=self.device).detach()
        self.reward_discount_factor = float(reward_discount_factor)
        self.num_policy_epochs = int(num_policy_epochs)
        self.sample_from_action_distributions = bool(sample_from_action_distributions)
        self.policy_batch_size = int(policy_batch_size)
        self.visualization_interval = int(visualization_interval)
        self.rewards = DEFAULT_REWARDS_MINIMAL if rewards is None else rewards
        self.use_supervision = use_supervision
        self.gradient_clipping_norm = gradient_clipping_norm
        # self.scheduler set by TorchTrainer superclass

        self.spawn_environments()

    def spawn_environments(self):
        global log_debug_indent
        log_debug('entered spawn_environments')
        log_debug_indent += 1

        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing)
        self.test_loader = self.dataloader.get_testing_dataloader(batch_size=self.batch_size,
                                                                  preprocessing=self.preprocessing)

        envs = []
        for loader, offset, batch_size, num_envs in [(self.train_loader, 0, self.batch_size, self.batch_size),
                                                     (self.test_loader, self.dataloader.get_dataset_sizes(self.split)[0],
                                                      self.batch_size, self.batch_size)]:
            all_env_img_paths = [[] for _ in range(num_envs)]
            all_env_gt_paths = [[] for _ in range(num_envs)]
            all_env_opt_brush_radius_paths = [[] for _ in range(num_envs)] if self.use_supervision else None
            all_env_non_max_suppressed_paths = [[] for _ in range(num_envs)] if self.use_supervision else None
            all_dl_sample_idxs = [[] for _ in range(num_envs)]

            for (_, _, _idxs) in loader:
                # needed for padding, to keep all environments synched
                if len(_idxs) < batch_size:
                    idxs = torch.cat((_idxs, torch.tensor([0 for _ in range(batch_size - len(_idxs))])), axis=0)
                else:
                    idxs = _idxs
                
                for env_idx, sample_idx in enumerate(idxs):
                    img_path = self.dataloader.training_img_paths[sample_idx]  # no need to add offset
                    gt_path = self.dataloader.training_gt_paths[sample_idx]  # no need to add offset
                    all_env_img_paths[env_idx].append(img_path)
                    all_env_gt_paths[env_idx].append(gt_path)
                    all_dl_sample_idxs[env_idx].append(sample_idx)
                    if self.use_supervision:
                        opt_brush_radius_path =\
                            os.path.join(os.path.dirname(os.path.dirname(img_path)), 'opt_brush_radius',
                                                         os.path.basename(img_path).replace('.png', '.pkl'))
                        non_max_supp_path =\
                            os.path.join(os.path.dirname(os.path.dirname(img_path)), 'non_max_suppressed',
                                                         os.path.basename(img_path).replace('.png', '.pkl'))
                        all_env_opt_brush_radius_paths[env_idx].append(opt_brush_radius_path)
                        all_env_non_max_suppressed_paths[env_idx].append(non_max_supp_path)
        
            env_list = [lambda img_paths=img_paths, gt_paths=gt_paths, opt_brush_radius_paths=opt_brush_radius_paths,
                        non_max_suppressed_paths=non_max_suppressed_paths:\
                        SegmentationEnvironmentMinimal(img_paths, gt_paths, self.model.patch_size,
                                                    self.train_loader.img_val_min, self.train_loader.img_val_max,
                                                    is_supervised=self.use_supervision, rewards=self.rewards,
                                                    supervision_optimal_brush_radius_map_paths=opt_brush_radius_paths,
                                                    supervision_non_max_suppressed_map_paths=non_max_suppressed_paths,
                                                    sample_preprocessing=self.preprocessing,
                                                    dl_sample_idxs=dl_sample_idxs)
                        for img_paths, gt_paths, opt_brush_radius_paths, non_max_suppressed_paths, dl_sample_idxs
                        in zip(all_env_img_paths, all_env_gt_paths, all_env_opt_brush_radius_paths,
                               all_env_non_max_suppressed_paths, all_dl_sample_idxs)]
            env = SubprocVecEnvTorch(env_list)
            envs.append(env)

        self.train_env = envs[0]
        self.test_env = envs[1]

        log_debug_indent -= 1
        log_debug('left spawn_environments')

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
            global log_debug_indent
            log_debug('entered on_train_batch_end')
            log_debug_indent += 1

            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                log_debug(f'on_train_batch_end: will evaluate (self.iteration_idx: {self.iteration_idx})')
                log_debug_indent += 1

                precision, recall, self.f1_score, reward_info = self.trainer.get_precision_recall_F1_score_validation()

                # {'reward_stats_first_quantities': reward_stats_first_quantities,
                #       'reward_stats_first_sums': reward_stats_first_sums,
                #       'reward_stats_sum_quantities': reward_stats_sum_quantities,
                #       'reward_stats_sum_sums': reward_stats_sum_sums,
                #       'reward_stats_avg_quantities': reward_stats_avg_quantities,
                #       'reward_stats_avg_sums': reward_stats_avg_sums}

                reward_info_flattened = {'rl_' + reward_type + '_' + key: reward_info[reward_type][key]
                                         for key in reward_info[list(reward_info.keys())[0]].keys()
                                         for reward_type in reward_info.keys()}

                metrics = {'precision': precision, 'recall': recall, 'f1_score': self.f1_score, **reward_info_flattened}
                print('Metrics at aggregate iteration %i (ep. %i, ep.-it. %i): %s'
                      % (self.iteration_idx, self.epoch_idx, self.epoch_iteration_idx, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=self.iteration_idx)
                    if self.do_visualize:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx,
                                                         self.epoch_idx, self.epoch_iteration_idx)
                
                if self.trainer.do_checkpoint\
                   and self.iteration_idx % self.trainer.checkpoint_interval == 0\
                   and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                    self.trainer._save_checkpoint(self.trainer.model, self.epoch_idx,
                                                  self.epoch_iteration_idx, self.iteration_idx)

                log_debug_indent -= 1
                log_debug(f'on_train_batch_end: evaluation complete')

            self.iteration_idx += 1
            self.epoch_iteration_idx += 1
            log_debug_indent -= 1
            log_debug('left on_train_batch_end')
    
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
            global log_debug_indent
            log_debug('entered compute_accumulated_discounted_returns')
            log_debug_indent += 1

            # memory.push(observation, model_output, sampled_actions,
            #                   self.terminated, reward, torch.tensor(float('nan')))
            for step_idx in reversed(range(len(self.memory))):  # differs from tutorial code (but appears to be equivalent)
                terminated = self.memory[step_idx][3]
                future_return = self.memory[step_idx + 1][-1] if step_idx < len(self.memory) - 1 else 0
                current_reward = self.memory[step_idx][4]
                self.memory[step_idx][-1] = current_reward + future_return * gamma * (1 - terminated)

            log_debug_indent -= 1
            log_debug('left compute_accumulated_discounted_returns')


        def __len__(self):
            return len(self.memory)

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path, iteration_index, epoch_idx, epoch_iteration_idx):
        global log_debug_indent
        log_debug('entered create_visualizations')
        log_debug_indent += 1

        # do not employ the random visualization scheme here, else the environments will get out of sync

        num_to_visualize = self.num_samples_to_visualize
        
        predictions_nstepwise_rgb_test_set = []
        positions_nstepwise_test_set = []
        
        longest_animation_length = 0

        larger_batch_size_test_loader = DataLoader(self.test_loader.dataset, batch_size=self.batch_size, shuffle=False)

        for (_batch_xs, batch_ys, batch_idxs) in larger_batch_size_test_loader:
            if num_to_visualize <= 0:
                self.test_env.reset()  # step to next
                continue
            else:
                if len(_batch_xs) > num_to_visualize:
                    _batch_xs = _batch_xs[:num_to_visualize]
                    batch_ys = batch_ys[:num_to_visualize]
                    batch_idxs = batch_idxs[:num_to_visualize]
                num_to_visualize -= len(_batch_xs)


            log_debug('create_visualizations: sampled new batch (batch_xs, batch_ys) from subset_dl')
            log_debug_indent += 1
        
            # batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.numpy()
            # output = self.model(batch_xs)
            # if type(output) is tuple:
            #     output = output[0]
            # preds = (output >= self.segmentation_threshold).float().cpu().detach().numpy()

            # preds_batch = []
            
            if self.use_supervision:
                batch_xs, opt_brush_radii, non_max_suppressed =\
                    _batch_xs[:, :3, :, :], _batch_xs[:, 3, :, :], _batch_xs[:, 4, :, :]
            else:
                batch_xs = _batch_xs
                opt_brush_radii, non_max_suppressed = None, None
            
            # sample_y = batch_ys[idx]
            # sample_x, sample_y = sample_x.to(self.device, dtype=torch.long), sample_y.to(self.device, dtype=torch.long)

            batch_xs, batch_ys = batch_xs.to(self.device, dtype=torch.long), batch_ys.to(self.device, dtype=torch.long)
            
            # batch_ys must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            if len(batch_ys.shape) == 4:
                batch_ys = torch.squeeze(batch_ys, axis=1)
            
            # no memory used here
            predictions_nstepwise, positions_nstepwise, reward, info =\
                self.trajectory_step(self.test_env, sample_from_action_distributions=self.sample_from_action_distributions,
                                     memory=None, visualization_interval=self.visualization_interval)

            # environment reset in trajectory_step

            for env_idx in filter(lambda i: i >= 0, [env_idx if sample_idx in batch_idxs else -1 for env_idx, sample_idx in enumerate(info['info_sample_idx'])]):
                # stack all animation frames, dim(nsteps, height, width)
                predictions_nstepwise_list = [predictions_nstepwise[timestep_idx][env_idx].cpu().numpy().astype(int)
                                              for timestep_idx in range(len(predictions_nstepwise))]                
                predictions_nstepwise_np = np.stack(predictions_nstepwise_list)
                
                sample_y = batch_ys[env_idx]
                sample_y_np = sample_y.cpu().numpy()
                
                merged = np.expand_dims(2 * predictions_nstepwise_np + sample_y_np, axis=0)
                green_channel = merged == 3  # true positives
                red_channel = merged == 2  # false positive
                blue_channel = merged == 1  # false negative
                rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=0) # shape(rgb, nsteps, height, width)

                predictions_nstepwise_rgb_test_set.append(rgb)

                positions_nstepwise_list = [positions_nstepwise[timestep_idx][env_idx]
                                            for timestep_idx in range(len(positions_nstepwise))]
                positions_nstepwise_test_set.append(positions_nstepwise_list)
                
                if len(predictions_nstepwise_list) > longest_animation_length:
                    longest_animation_length = len(predictions_nstepwise)


                # preds = envs.get_unpadded_segmentation().float().detach().cpu()
                # preds_batch.append(preds)

            # for idx, _sample_x in enumerate(batch_xs):
            #     log_debug(f'create_visualizations: processing sample {idx} in batch')
            #     log_debug_indent += 1

            #     if self.use_supervision:
            #         sample_x, opt_brush_radii, non_max_suppressed = _sample_x[:3], _sample_x[3], _sample_x[4]
            #     else:
            #         sample_x = _sample_x
            #         opt_brush_radii, non_max_suppressed = None, None
            #     sample_y = batch_ys[idx]
            #     sample_x, sample_y = sample_x.to(self.device, dtype=torch.long), sample_y.to(self.device, dtype=torch.long)
                
            #     # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            #     # accordingly, sample_y must be of shape (H, W) not (1, H, W)
            #     sample_y = torch.squeeze(sample_y)

            #     # TODO: do not pass opt_brush_radii and non_max_suppressed, else environment will assume we're working off-policy
            #     env = SegmentationEnvironmentMinimal(sample_x, sample_y, self.model.patch_size,
            #                                          self.test_loader.img_val_min, self.test_loader.img_val_max,
            #                                          is_supervised=self.use_supervision, rewards=self.rewards,
            #                                          supervision_optimal_brush_radius_map=opt_brush_radii,
            #                                          supervision_non_max_suppressed_map=non_max_suppressed)
            #     env.reset()
                
            #     # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
            #     # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
            #     # loop until termination/timeout already inside this function
            #     predictions_nstepwise, positions_nstepwise, reward, info =\
            #         self.trajectory_step(env, sample_from_action_distributions=self.sample_from_action_distributions, visualization_interval=self.visualization_interval)


            #     predictions_nstepwise_np = np.stack(predictions_nstepwise)  # stack all animation frames, dim(nsteps, height, width)
                

            #     sample_y_np = sample_y.cpu().numpy()
                
            #     merged = np.expand_dims(2 * predictions_nstepwise_np + sample_y_np, axis=0)
            #     green_channel = merged == 3  # true positives
            #     red_channel = merged == 2  # false positive
            #     blue_channel = merged == 1  # false negative
            #     rgb = np.concatenate((red_channel, green_channel, blue_channel), axis=0) # shape(rgb, nsteps, height, width)

            #     predictions_nstepwise_rgb_test_set.append(rgb)
            #     positions_nstepwise_test_set.append(positions_nstepwise)
                
            #     if len(predictions_nstepwise) > longest_animation_length:
            #         longest_animation_length = len(predictions_nstepwise)

            #     preds = env.get_unpadded_segmentation().float().detach().cpu()
            #     preds_batch.append(preds)

            #     log_debug_indent -= 1

            log_debug_indent -= 1

        log_debug(f'create_visualizations: all batches processed; generating GIF; longest_animation_length is {longest_animation_length}')

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

            log_debug(f'create_visualizations: generating GIF, looping over timesteps...')

            gif_frames = []
            for timestep_idx in range(longest_animation_length):
            # uncomment to spam output console  
                # log_debug('.', no_line_break=True, log_timestamp=False)

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
            
            log_debug('create_visualizations: left loop')
            log_debug('create_visualizations: saving GIF')

            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=100, loop=0) # duration is milliseconds between frames, loop=0 means loop infinite times

            log_debug('create_visualizations: GIF saved')
            
            # if not working, convert via: imageio.core.util.Array(numpy_array)

            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            # self._fill_images_array(torch.stack(preds_batch, dim=0).unsqueeze(1).float().detach().cpu().numpy(), batch_ys.numpy(), images)

        # return super().create_visualizations(file_path)
        
        log_debug_indent -= 1
        log_debug('left create_visualizations')
        return gif_path

    def _train_step(self, model, device, train_loader, callback_handler):
        global log_debug_indent
        log_debug('entered _train_step')
        log_debug_indent += 1

        # WARNING: some models subclassing TorchTrainer overwrite this function, so make sure any changes here are
        # reflected appropriately in these models' files
        
        # TODO: parallelize with SubprocVecEnv

        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0

        for (_xs, ys, idxs) in train_loader:
            log_debug('_train_step: sampled new batch (xs, ys) from train_loader')
            log_debug_indent += 1

            _xs, ys = _xs.to(device, dtype=torch.float32), ys.to(device, dtype=torch.long)

            xs, opt_brush_radii, non_max_suppressed = _xs[:, :3], _xs[:, 3], _xs[:, 4]

            # batch_ys must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            if len(ys.shape) == 4:
                ys = torch.squeeze(ys, axis=1)
            
            # already called in environments:
            # envs.reset()
            
            if self.use_supervision:
                memory = self.ReplayMemory(capacity=int(self.replay_memory_capacity) * xs.shape[0],
                                           balancing_trajectory_length=int(self.replay_memory_capacity) * xs.shape[0])
            else:
                # need to be able to compute returns individually
                memory = [self.ReplayMemory(capacity=int(self.replay_memory_capacity),
                                            balancing_trajectory_length=int(self.replay_memory_capacity))
                          for _ in xs.shape[0]]

            # if we exit this function and "terminated" is false, we timed out
            # (more than self.max_rollout_len iterations)
            # loop until termination/timeout already inside this function
            self.trajectory_step(self.train_env, sample_from_action_distributions=self.sample_from_action_distributions,
                                 memory=memory)
            # environment reset in trajectory_step

            if not self.use_supervision:
                for _memory in memory:
                    _memory.compute_accumulated_discounted_returns(gamma=self.reward_discount_factor)
        
            memory_list_to_sample_from = None

            if self.use_supervision:
                memory_list_to_sample_from = memory.memory
            else:
                memory_list_to_sample_from = []
                for m in memory:
                    memory_list_to_sample_from.extend(m.memory)

            for epoch_idx in range(self.num_policy_epochs):
                log_debug(f'_train_step: entered policy epoch {epoch_idx}')
                log_debug_indent += 1

                policy_batch = random.sample(memory_list_to_sample_from,
                                            min(self.policy_batch_size, len(memory_list_to_sample_from)))

                # memory.push(observation, model_output, action_log_probabilities, sampled_actions,
                #                   self.terminated, reward, torch.tensor(float('nan')))

                # model_output_sample, action_log_probabilities_sample, sampled_actions_sample: have requires_grad=True
                
                observation_sample = torch.stack([policy_sample[0] for policy_sample in policy_batch])
                model_output_sample = torch.stack([policy_sample[1] for policy_sample in policy_batch])

                # unused:
                # unnormalized_actions_sample = torch.stack([policy_sample[2] for policy_sample in policy_batch])
                # terminated_sample = torch.stack([policy_sample[3] for policy_sample in policy_batch])
                # reward_sample = torch.stack([policy_sample[4] for policy_sample in policy_batch])

                returns_sample = torch.stack([policy_sample[5] for policy_sample in policy_batch])
                supervision_info = torch.stack([policy_sample[6] for policy_sample in policy_batch])

                # recalculate log probabilities (see tutorial: also recalculated)
                # if we do not recalculate, we will get errors related to repeated backpropagation

                # model_output = self.model(observation_sample.unsqueeze(0).unsqueeze(0))
                model_output = self.model(observation_sample)

                # mean = alpha / (alpha + beta)

                # alphas, betas = TorchRLTrainer.get_beta_params(mu=model_output[0], sigma=self.std)  # remove batch dimension, multiply and add to avoid extreme points
                # print(f'alphas, betas: {(alphas, betas)}')
                
                # dist = torch.distributions.Beta(alphas, betas)
                
                if self.use_supervision:
                    # loss_function expects args: input (model output), target
                    policy_loss = self.loss_function(self.unnormalize_model_output(model_output, min_patch_size=min(list(self.model.patch_size))), supervision_info)
                else:
                    covariance_matrix = torch.diag(self.std)
                    dist = torch.distributions.MultivariateNormal(loc=model_output[0], covariance_matrix=covariance_matrix)
                    action_log_probabilities_sample = dist.log_prob(model_output_sample)
                    policy_loss = -(action_log_probabilities_sample * returns_sample).mean()
                # TODO: add entropy loss and see if it helps
                # entropy_loss = ...
                    
                opt.zero_grad()
                loss = policy_loss
                loss.backward(retain_graph=True) # retain computational graph because needed when sampling the same trajectory multiple time
                
                if self.gradient_clipping_norm not in [None, 0.0]:
                    for param in self.model.parameters():  # gradient clipping
                        param.grad.data.clamp_(-self.gradient_clipping_norm, self.gradient_clipping_norm)
                opt.step()
                
                with torch.no_grad():
                    train_loss += policy_loss.item()

                log_debug_indent -= 1

            ##########

            # for idx, _sample_x in enumerate(xs):
            #     log_debug(f'_train_step: processing sample {idx} in batch')

            #     sample_x, opt_brush_radii, non_max_suppressed = _sample_x[:3], _sample_x[3], _sample_x[4]

            #     log_debug_indent += 1
            #     sample_y = ys[idx]
                
            #     sample_x, sample_y = sample_x.to(device, dtype=torch.float32), sample_y.to(device, dtype=torch.long)
            #     # sample_y = sample_y.to(device, dtype=torch.long)

            #     # insert each patch into ReplayMemory
            #     # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            #     # accordingly, sample_y must be of shape (H, W) not (1, H, W)
            #     sample_y = torch.squeeze(sample_y)
            #     env = SegmentationEnvironmentMinimal(sample_x, sample_y, self.model.patch_size,
            #                                          train_loader.img_val_min, train_loader.img_val_max,
            #                                          is_supervised=self.use_supervision, rewards=self.rewards,
            #                                          supervision_optimal_brush_radius_map=opt_brush_radii,
            #                                          supervision_non_max_suppressed_map=non_max_suppressed)
            #     # using maximum memory capacity for balancing_trajectory_length is suboptimal;
            #     # could e.g. empirically approximate expected trajectory length and use that instead
            #     memory = self.ReplayMemory(capacity=int(self.replay_memory_capacity),
            #                                balancing_trajectory_length=int(self.replay_memory_capacity))
                
            #     # "state" is input to model
            #     # observation in init state: RGB (3), history (5 by default), brush state (1)
                
            #     # env.step returns: (new_observation, reward, done, new_info)
            #     # "state" in tutorial is "observation" here
                
            #     # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
            #     # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate

            #     # if we exit this function and "terminated" is false, we timed out
            #     # (more than self.max_rollout_len iterations)
            #     # loop until termination/timeout already inside this function
            #     self.trajectory_step(env, sample_from_action_distributions=self.sample_from_action_distributions,
            #                          memory=memory)

            #     if not self.use_supervision:
            #         memory.compute_accumulated_discounted_returns(gamma=self.reward_discount_factor)

            #     for epoch_idx in range(self.num_policy_epochs):
            #         log_debug(f'_train_step: entered policy epoch {epoch_idx}')
            #         log_debug_indent += 1

            #         loss_accumulator = []
            #         policy_batch = memory.sample(self.policy_batch_size)
            #         for sample_idx, sample in enumerate(policy_batch):
            #             log_debug(f'_train_step: processing sample {sample_idx} in policy epoch {epoch_idx}')
            #             log_debug_indent += 1

            #             # memory.push(observation, model_output, action_log_probabilities, sampled_actions,
            #             #                   self.terminated, reward, torch.tensor(float('nan')))

            #             # model_output_sample, action_log_probabilities_sample, sampled_actions_sample: have requires_grad=True
                        
            #             if self.use_supervision:
            #                 observation_sample, model_output_sample,\
            #                     unnormalized_actions_sample, terminated_sample, reward_sample, returns_sample, supervision_info = sample
            #             else:
            #                 observation_sample, model_output_sample,\
            #                     unnormalized_actions_sample, terminated_sample, reward_sample, returns_sample = sample

            #             # recalculate log probabilities (see tutorial: also recalculated)
            #             # if we do not recalculate, we will get errors related to repeated backpropagation

            #             # model_output = self.model(observation_sample.unsqueeze(0).unsqueeze(0))
            #             model_output = self.model(observation_sample.unsqueeze(0))

            #             # mean = alpha / (alpha + beta)

            #             # alphas, betas = TorchRLTrainer.get_beta_params(mu=model_output[0], sigma=self.std)  # remove batch dimension, multiply and add to avoid extreme points
            #             # print(f'alphas, betas: {(alphas, betas)}')
                        
            #             # dist = torch.distributions.Beta(alphas, betas)
                        
            #             if self.use_supervision:
            #                 # loss_function expects args: input (model output), target
            #                 policy_loss = self.loss_function(self.unnormalize_model_output(model_output, env), supervision_info)
            #             else:
            #                 covariance_matrix = torch.diag(self.std)
            #                 dist = torch.distributions.MultivariateNormal(loc=model_output[0], covariance_matrix=covariance_matrix)
            #                 action_log_probabilities_sample = dist.log_prob(model_output_sample)
            #                 policy_loss = -(action_log_probabilities_sample * returns_sample).mean()
            #             # TODO: add entropy loss and see if it helps
            #             # entropy_loss = ...
                            
            #             opt.zero_grad()
            #             loss = policy_loss
            #             loss.backward(retain_graph=True) # retain computational graph because needed when sampling the same trajectory multiple time
                        
            #             if self.gradient_clipping_norm not in [None, 0.0]:
            #                 for param in self.model.parameters():  # gradient clipping
            #                     param.grad.data.clamp_(-self.gradient_clipping_norm, self.gradient_clipping_norm)
            #             opt.step()
                        
            #             with torch.no_grad():
            #                 train_loss += policy_loss.item()
                            
            #             log_debug_indent -= 1
            
            #         log_debug_indent -= 1

            #     log_debug_indent -= 1

            #     # code for one sample ends here
            
            log_debug_indent -= 1
            callback_handler.on_train_batch_end()
            
            train_loss /= (len(train_loader.dataset) * self.policy_batch_size * self.num_policy_epochs)
            callback_handler.on_epoch_end()
            try:
                self.scheduler.step()
            except:
                self.scheduler.step(train_loss)  # TODO: perform step in _eval_step and use validation loss instead of train loss
        
        log_debug_indent -= 1
        log_debug('left _train_step')
        return train_loss

    def _eval_step(self, model, device, test_loader):
        # it does not make sense to have this function in an RL trainer
        return 0.0

        # global log_debug_indent

        # log_debug('entered _eval_step')
        # log_debug_indent += 1

        # if self.loss_function is None:
        #     log_debug_indent -= 1
        #     log_debug('left _eval_step because self.loss_function is None')
        #     return 0.0

        # model.eval()
        # test_loss = 0

        # for (xs, ys) in test_loader:
        #     log_debug('_eval_step: sampled batch (xs, ys) from test_loader')
        #     log_debug_indent += 1

        #     for idx, sample_x in enumerate(xs):
        #         log_debug(f'_train_step: processing sample {idx}')
        #         log_debug_indent += 1

        #         sample_y = sample_y[idx]
        #         sample_x, sample_y = sample_x.to(device, dtype=torch.long), sample_y.to(device, dtype=torch.long)
        #         # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
        #         # accordingly, sample_y must be of shape (H, W) not (1, H, W)
        #         sample_y = torch.squeeze(sample_y)
        #         env = SegmentationEnvironmentMinimal(sample_x, sample_y, self.model.patch_size,
        #                                              self.test_loader.img_val_min, self.test_loader.img_val_max,
        #                                              is_supervised=self.use_supervision) # take standard reward

        #         # "state" is input to model
        #         # observation in init state: RGB (3), history (5 by default), brush state (1)
        #         # env.step returns: (new_observation, reward, done, new_info)
        #         # "state" in tutorial is "observation" here
                
        #         # in https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html,
        #         # they loop for "rollouts.rollout_size" iterations, but here, we loop until network tells us to terminate
        #         # loop until termination/timeout already inside this function
        #         predictions_nsteps, positions_nsteps, reward, info = self.trajectory_step(env, sample_from_action_distributions=False)
        #         preds = env.get_unpadded_segmentation().float()
        #         test_loss += self.loss_function(preds, sample_y).item()
                
        #         log_debug_indent -= 1

        #     log_debug_indent -= 1
        
        # test_loss /= len(test_loader.dataset)
        # log_debug_indent -= 1
        # log_debug('left _eval_step')
        # return test_loss

    def unnormalize_model_output(self, model_output, min_patch_size):
        if self.use_supervision:
            pre = torch.stack([-1 + 2 * torch.pi * model_output[..., 0],  # angle in [-pi, pi]
                                model_output[..., 1] * (min_patch_size / 2),  # brush_radius in [0, min_patch_size / 2]
                                model_output[..., 2] * (min_patch_size / 2)])  # magnitude in [0, min_patch_size / 2]
        else:
            pre = torch.stack([-1 + 2 * torch.pi * model_output[..., 0],  # delta_angle in [-pi, pi]
                                torch.round(-1 + 2 * model_output[..., 1])])  # new_brush_state
        if len(model_output.shape) == 1:
            return pre
        else:
            return torch.permute(pre, (1, 0))

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
            memory (Memory or list of Memory, optional): memory/memories to store the trajectories to during training. Defaults to None.
            visualization_interval (int, optional): number of timesteps between frames of the returned list of observations
        Returns:
            list of observations that can be used to generate animations of the agent's trajectory, or [] if visualization_interval <= 0 (List of torch.Tensor)
        """
        global log_debug_indent
        log_debug(f'entered trajectory_step')
        log_debug_indent += 1

        # here, the reward information is aggregated over the *timesteps* (sum and average)

        is_multi_env = isinstance(env, SubprocVecEnvTorch)

        # !!!!!!!!!!
        # WARNING: this was commented and changed

        # predictions_nsteps = [env.get_unpadded_segmentation().float().detach().cpu().numpy()]
        # positions_nsteps = [env.get_rounded_agent_pos()]

        predictions_nsteps = []
        positions_nsteps = []

        # info_sum = [{'reward_decomp_quantities': {}, 'reward_decomp_sums': {}}]
        # moved into environment

        # !!!!!!!!!!

        # if self.use_supervision and memory is not None:
        #     # initialization of exploration policy
        #     # (as in walker.py)

        #     dims = env.img_size
        #     y, x = float(random.randint(0, dims[0]-1)), float(random.randint(0, dims[1]-1))

        #     last_y_int, last_x_int = int(np.round(y)), int(np.round(x))
            
        #     is_on_road = env.gt[last_y_int, last_x_int] > 0
        #     is_on_max = env.supervision_non_max_suppressed_map[last_y_int, last_x_int] > 0
            
        #     policy_state = {'is_on_road': is_on_road,
        #                     'is_on_max': is_on_max,
        #                     'last_angle': 0.0,
        #                     'last_x_int': last_x_int,
        #                     'last_y_int': last_y_int,
        #                     'current_is_visited': False,
        #                     'visited': torch.zeros(dims, device=env.img.device, dtype=torch.int8)}
        # else:
        #     policy_state = None

        observation, *_, info = env.step(torch.stack([SegmentationEnvironmentMinimal.get_neutral_action()
                                                      for _ in range(env.nenvs)])
                                         if is_multi_env else SegmentationEnvironmentMinimal.get_neutral_action())

        unpadded_segmentation = [info[idx]['unpadded_segmentation'] for idx in range(env.nenvs)] if is_multi_env else info['unpadded_segmentation']
        rounded_agent_pos = [info[idx]['rounded_agent_pos'] for idx in range(env.nenvs)] if is_multi_env else info['rounded_agent_pos']

        predictions_nsteps.append(unpadded_segmentation)
        positions_nsteps.append(rounded_agent_pos)

        # predictions_nsteps.append(env.get_unpadded_segmentation().float().detach().cpu().numpy())
        # positions_nsteps.append(env.get_rounded_agent_pos())

        if sample_from_action_distributions is None:
            sample_from_action_distributions = self.sample_from_action_distributions

        # log_debug(f'looping over range(self.rollout_len) == range({self.rollout_len})', no_line_break=True)

        log_debug(f'looping over range(self.rollout_len) == range({self.rollout_len})...')

        # stores the segmentation states and agent positions seen before termination in each environment
        pre_termination_unpadded_segmentation = unpadded_segmentation
        pre_termination_rounded_agent_pos = rounded_agent_pos

        for timestep_idx in range(self.rollout_len):
            # uncomment to spam output console  
            # log_debug(f'.', no_line_break=True, log_timestamp=False)

            if len(observation.shape) == 3:
                model_output = self.model(observation.detach().unsqueeze(0)).detach()
            else:
                model_output = self.model(observation.detach()).detach()

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
                action = torch.normal(mean=model_output, std=self.std)
                action = torch.clamp(action, 0.0, 1.0)
                if len(observation.shape) == 3:
                    action = action[0]
            else:
                if len(observation.shape) == 3:
                    action = model_output.squeeze()  # remove batch dimension

            model_output_unnormalized = self.unnormalize_model_output(model_output, min(list(self.model.patch_size)))

            new_observation, reward, terminated, info = env.step(model_output_unnormalized)
            if not self.use_supervision:
                terminated = torch.Tensor([timestep_idx == self.rollout_len-1] * env.nenvs
                                           if is_multi_env else [timestep_idx == self.rollout_len-1]).to(self.device)
                if not terminated:
                    pre_termination_unpadded_segmentation = [info[idx]['unpadded_segmentation'] for idx in range(env.nenvs)] if is_multi_env else info['unpadded_segmentation']
                    pre_termination_rounded_agent_pos = [info[idx]['rounded_agent_pos'] for idx in range(env.nenvs)] if is_multi_env else info['rounded_agent_pos']
            else:
                if is_multi_env:
                    for env_idx in range(env.nenvs):
                        if not terminated[env_idx]:
                            pre_termination_unpadded_segmentation[env_idx] = info[env_idx]['unpadded_segmentation']
                            pre_termination_rounded_agent_pos[env_idx] = info[env_idx]['rounded_agent_pos']
                else:
                    if not terminated:
                        pre_termination_unpadded_segmentation = info['unpadded_segmentation']
                        pre_termination_rounded_agent_pos = info['rounded_agent_pos']
                
                if torch.all(terminated):  # don't waste time
                    break

            if memory is not None:
                # still use one memory, even if we have multiple environments!
                # WARNING: memory length must be adapted to batch size!
                if is_multi_env:
                    for env_idx in range(env.nenvs):
                        # do not push observation to memory if this environment has already terminated
                        # reconsider? 
                        if terminated[env_idx]:
                            continue

                        env_memory = memory if not isinstance(memory, list) else memory[env_idx]
                        if 'supervision_desired_outputs' in info[env_idx]:
                            env_memory.push(observation[env_idx], model_output[env_idx], model_output_unnormalized[env_idx],
                            torch.tensor(terminated[env_idx]), reward[env_idx], torch.tensor(float('nan')),
                            info[env_idx]['supervision_desired_outputs'])
                        elif not self.use_supervision:
                            env_memory.push(observation[env_idx], model_output[env_idx], model_output_unnormalized[env_idx],
                            torch.tensor(terminated[env_idx]), reward[env_idx], torch.tensor(float('nan')))
                else:
                    if 'supervision_desired_outputs' in info:
                        # "supervision_desired_outputs" is tensor with entries "angle", "brush_radius", "magnitude"
                        # (with values normed to be between 0 and 1, so that we can directly apply a loss function between
                        #  the policy network's outputs and the desired outputs)
                        memory.push(observation, model_output, model_output_unnormalized, torch.tensor(terminated), reward, torch.tensor(float('nan')), info['supervision_desired_outputs'])
                    elif not self.use_supervision:
                        memory.push(observation, model_output, model_output_unnormalized, torch.tensor(terminated), reward, torch.tensor(float('nan')))

            observation = new_observation
            
            if visualization_interval > 0 and timestep_idx % visualization_interval == 0:
                unpadded_segmentation = [info[idx]['unpadded_segmentation'] for idx in range(env.nenvs)] if is_multi_env else info['unpadded_segmentation']
                rounded_agent_pos = [info[idx]['rounded_agent_pos'] for idx in range(env.nenvs)] if is_multi_env else info['rounded_agent_pos']

                predictions_nsteps.append(unpadded_segmentation)
                positions_nsteps.append(rounded_agent_pos)
        
        log_debug('left loop')

        # append last frame for metric calculation

        if visualization_interval <= 0:
            predictions_nsteps.append(pre_termination_unpadded_segmentation)
            positions_nsteps.append(pre_termination_rounded_agent_pos)


        if is_multi_env:
            info_sample_idx = [info[idx]['sample_idx'] for idx in range(env.nenvs)]
            info_sum = [info[idx]['info_sum'] for idx in range(env.nenvs)]
            info_avg = []
            for env_idx in range(env.nenvs):
                new_info_avg = {'reward_decomp_quantities': {}, 'reward_decomp_sums': {}}
                # timestep_idx + 1 due to zero-based indexing; add another 1 due to initial timestep 
                new_info_avg['reward_decomp_quantities'] = {k: v / (timestep_idx + 2) for k, v in info_sum[env_idx]['reward_decomp_quantities'].items()}
                new_info_avg['reward_decomp_sums'] = {k: v / (timestep_idx + 2) for k, v in info_sum[env_idx]['reward_decomp_sums'].items()}

                info_avg.append(new_info_avg)
        else:
            info_sample_idx = info['sample_idx']
            info_sum = info['info_sum']
            info_avg = {'reward_decomp_quantities': {}, 'reward_decomp_sums': {}}
            # timestep_idx + 1 due to zero-based indexing; add another 1 due to initial timestep 
            info_avg['reward_decomp_quantities'] = {k: v / (timestep_idx + 2) for k, v in info_sum['reward_decomp_quantities'].items()}
            info_avg['reward_decomp_sums'] = {k: v / (timestep_idx + 2) for k, v in info_sum['reward_decomp_sums'].items()}

        log_debug_indent -= 1
        log_debug('left trajectory_step')

        # move to next sample
        env.reset()

        return predictions_nsteps, positions_nsteps, reward, {'info_timestep_sum': info_sum,
                                                              'info_timestep_avg': info_avg,
                                                              'info_sample_idx': info_sample_idx}

    def get_F1_score_validation(self):
        _, _, f1_score, _ = self.get_precision_recall_F1_score_validation()
        return f1_score

    def get_precision_recall_F1_score_validation(self):
        global log_debug_indent
        log_debug('entered get_precision_recall_F1_score_validation')
        log_debug_indent += 1
        
        # this function also returns reward statistics (averaged, summed, and for the first sample)

        reward_stats__first_sample__timestep_sum__reward_quantities = {}
        reward_stats__first_sample__timestep_sum__reward_sums = {}
        
        reward_stats__first_sample__timestep_avg__reward_quantities = {}
        reward_stats__first_sample__timestep_avg__reward_sums = {}

        reward_stats__sample_sum__timestep_sum__reward_quantities = {}
        reward_stats__sample_sum__timestep_sum__reward_sums = {}
        
        reward_stats__sample_sum__timestep_avg__reward_quantities = {}
        reward_stats__sample_sum__timestep_avg__reward_sums = {}

        self.model.eval()
        precisions, recalls, f1_scores = [], [], []
        num_samples = 0

        # create new test loader with larger batch size to speed up inference

        larger_batch_size_test_loader = DataLoader(self.test_loader.dataset, batch_size=self.batch_size, shuffle=False)

        for (_batch_xs, batch_ys, batch_idxs) in larger_batch_size_test_loader:
            log_debug(f'get_precision_recall_F1_score_validation: sampled new batch (xs, ys) from self.test_loader (size: {_batch_xs.shape[0]})')
            log_debug_indent += 1

            if self.use_supervision:
                batch_xs, opt_brush_radii, non_max_suppressed =\
                    _batch_xs[:, :3, :, :], _batch_xs[:, 3, :, :], _batch_xs[:, 4, :, :]
            else:
                batch_xs = _batch_xs
                opt_brush_radii, non_max_suppressed = None, None

            batch_xs, batch_ys = batch_xs.to(self.device, dtype=torch.long), batch_ys.to(self.device, dtype=torch.long)
            
            # batch_ys must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            if len(batch_ys.shape) == 4:
                batch_ys = torch.squeeze(batch_ys, axis=1)

            # no memory used here

            predictions_nstepwise, positions_nstepwise, reward, info =\
                self.trajectory_step(self.test_env, sample_from_action_distributions=self.sample_from_action_distributions)
            
            # environment reset in trajectory_step

            for idx in filter(lambda i: i >= 0, [env_idx if sample_idx in batch_idxs else -1 for env_idx, sample_idx in enumerate(info['info_sample_idx'])]):
                info_timestep_sum = info['info_timestep_sum'][idx] if 'info_timestep_sum' in info else {}
                info_timestep_avg = info['info_timestep_avg'][idx] if 'info_timestep_sum' in info else {}

                if 'reward_decomp_quantities' in info_timestep_sum:
                    for key in info_timestep_sum['reward_decomp_quantities'].keys():
                        reward_stats__sample_sum__timestep_sum__reward_quantities[key] = reward_stats__sample_sum__timestep_sum__reward_quantities.get(key, 0.0) + info_timestep_sum['reward_decomp_quantities'][key]
                        if idx == 0:
                            reward_stats__first_sample__timestep_sum__reward_quantities[key] = reward_stats__first_sample__timestep_sum__reward_quantities.get(key, 0.0) + info_timestep_sum['reward_decomp_quantities'][key]
                        
                if 'reward_decomp_sums' in info_timestep_sum:
                    for key in info_timestep_sum['reward_decomp_sums'].keys():
                        reward_stats__sample_sum__timestep_sum__reward_sums[key] = reward_stats__sample_sum__timestep_sum__reward_sums.get(key, 0.0) + info_timestep_sum['reward_decomp_sums'][key]
                        if idx == 0:
                            reward_stats__first_sample__timestep_sum__reward_sums[key] = reward_stats__first_sample__timestep_sum__reward_sums.get(key, 0.0) + info_timestep_sum['reward_decomp_sums'][key]

                if 'reward_decomp_quantities' in info_timestep_avg:
                    for key in info_timestep_avg['reward_decomp_quantities'].keys():
                        reward_stats__sample_sum__timestep_avg__reward_quantities[key] = reward_stats__sample_sum__timestep_avg__reward_quantities.get(key, 0.0) + info_timestep_avg['reward_decomp_quantities'][key]
                        if idx == 0:
                            reward_stats__first_sample__timestep_avg__reward_quantities[key] = reward_stats__first_sample__timestep_avg__reward_quantities.get(key, 0.0) + info_timestep_avg['reward_decomp_quantities'][key]
                        
                if 'reward_decomp_sums' in info_timestep_avg:
                    for key in info_timestep_avg['reward_decomp_sums'].keys():
                        reward_stats__sample_sum__timestep_avg__reward_sums[key] = reward_stats__sample_sum__timestep_avg__reward_sums.get(key, 0.0) + info_timestep_avg['reward_decomp_sums'][key]
                        if idx == 0:
                            reward_stats__first_sample__timestep_avg__reward_sums[key] = reward_stats__first_sample__timestep_avg__reward_sums.get(key, 0.0) + info_timestep_avg['reward_decomp_sums'][key]

                preds = predictions_nstepwise[-1][idx] # info[idx].get('unpadded_segmentation', {})
                precision, recall, f1_score = precision_recall_f1_score_torch(preds, batch_ys[idx])
                precisions.append(precision.cpu().numpy())
                recalls.append(recall.cpu().numpy())
                f1_scores.append(f1_score.cpu().numpy())

                # print(f'Reward information for sample {idx}: {info}')
                
                log_debug_indent -= 1

            log_debug_indent -= 1

        log_debug('get_precision_recall_F1_score_validation: all batches processed')

        reward_stats__sample_avg__timestep_sum__reward_quantities = {k: v / num_samples for k, v in reward_stats__sample_sum__timestep_sum__reward_quantities.items()}
        reward_stats__sample_avg__timestep_sum__reward_sums = {k: v / num_samples for k, v in reward_stats__sample_sum__timestep_sum__reward_sums.items()}
        
        reward_stats__sample_avg__timestep_avg__reward_quantities = {k: v / num_samples for k, v in reward_stats__sample_sum__timestep_avg__reward_quantities.items()}
        reward_stats__sample_avg__timestep_avg__reward_sums = {k: v / num_samples for k, v in reward_stats__sample_sum__timestep_avg__reward_sums.items()}

        reward_info = {'reward_stats__first_sample__timestep_sum__reward_quantities': reward_stats__first_sample__timestep_sum__reward_quantities,
                       'reward_stats__first_sample__timestep_sum__reward_sums': reward_stats__first_sample__timestep_sum__reward_sums,
                       'reward_stats__first_sample__timestep_avg__reward_quantities': reward_stats__first_sample__timestep_avg__reward_quantities,
                       'reward_stats__first_sample__timestep_avg__reward_sums': reward_stats__first_sample__timestep_avg__reward_sums,
                       'reward_stats__sample_sum__timestep_sum__reward_quantities': reward_stats__sample_sum__timestep_sum__reward_quantities,
                       'reward_stats__sample_sum__timestep_sum__reward_sums': reward_stats__sample_sum__timestep_sum__reward_sums,
                       'reward_stats__sample_sum__timestep_avg__reward_quantities': reward_stats__sample_sum__timestep_avg__reward_quantities,
                       'reward_stats__sample_sum__timestep_avg__reward_sums': reward_stats__sample_sum__timestep_avg__reward_sums,
                       'reward_stats__sample_avg__timestep_sum__reward_quantities': reward_stats__sample_avg__timestep_sum__reward_quantities,
                       'reward_stats__sample_avg__timestep_sum__reward_sums': reward_stats__sample_avg__timestep_sum__reward_sums,
                       'reward_stats__sample_avg__timestep_avg__reward_quantities': reward_stats__sample_avg__timestep_avg__reward_quantities,
                       'reward_stats__sample_avg__timestep_avg__reward_sums': reward_stats__sample_avg__timestep_avg__reward_sums}

        # aggregated reward information printed by evaluation logic calling this function
        # print(f'Aggregated reward information: {reward_info}')

        log_debug_indent -= 1
        log_debug('left get_precision_recall_F1_score_validation')
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), reward_info

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self, param)
                   for param in ['history_size', 'rollout_len', 'replay_memory_capacity', 'reward_discount_factor',
                                 'num_policy_epochs', 'policy_batch_size', 'sample_from_action_distributions',
                                 'visualization_interval', 'use_supervision', 'gradient_clipping_norm']
                   if hasattr(self, param)}),
                **({k: v.item() for k, v in zip(['std_delta_angle', 'std_brush_state'], self.std)}),
                **({'rl_' + k: v for k, v in self.rewards.items()}),
                **({param: getattr(self.model, param)
                   for param in ['patch_size']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr, model):
        return optim.Adam(model.parameters(), lr=lr)
