import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import pickle
import torch
import torch.nn.functional as F

from gym import Env, spaces
from utils import *


BRUSH_STATE_ERASE, BRUSH_STATE_NOTHING, BRUSH_STATE_PAINT = -1, 0, 1
TERMINATE_NO, TERMINATE_YES = 0, 1

# penalty for false negative should not be smaller than reward for true positive,
# else agent could paint a road spot, then erase it, then paint it again, etc.
# (not observed, but possible loophole)

DEFAULT_REWARDS = {
    'changed_brush_pen': 0.001,
    'changed_brush_rad_pen': 0.001,
    'changed_angle_pen': 0.001,
    'false_neg_seg_pen': 0.01,
    'false_pos_seg_pen': 0.01,
    'time_pen': 0.0005,
    'unseen_pix_pen': 0.02,
    'true_pos_seg_rew': 0.01,
    'true_neg_seg_rew': 0.01,
    'unseen_pix_rew': 0.0007,
}

DEFAULT_REWARDS_MINIMAL = {
    'true_seg_rew': 0.01,
    'unseen_pix_rew': 0.001
}


class SegmentationEnvironment(Env):
    """Environment for Reinforcement Learning Approach
    It calculates the next inputs for the model based on the models output, as well as calculates the rewards
    and penalties. In general it saves the state of the agent and environment. The differences to the minimal
    version, which you can find in the bottom half of this script, are marked with '<<< diff'
    Args:
        img (torch.Tensor): image to segment, dimension (img_size, img_channels)
        gt (torch.Tensor): ground-truth of image to segment, or None if no segmentation is to be performed
        patch_size (tuple of int): size of patch visible to agent at a given timestep
        history_size (int): how many previous predictions of the network should be fed into the network
        img_val_min (int or float): minimum value of a pixel in input image
        img_val_max (int or float): maximum value of a pixel in input image
        rewards (dictionary): a dictionary of rewards, penalties have a negative sign
    """
    def __init__(self, img, gt, patch_size, history_size, img_val_min, img_val_max, rewards=DEFAULT_REWARDS):
        super(SegmentationEnvironment, self).__init__()
        self.patch_size = patch_size
        self.img = img  # <<< diff
        self.gt = gt
        self.img_size = img.shape[1:]
        self.rewards = rewards
        self.device = self.img.device
        self.history_size = history_size  # <<< diff
        self.img_val_min = img_val_min  # <<< diff
        self.img_val_max = img_val_max  # <<< diff
        self.padding_value = -1 if img_val_min == 0 else -2 * img_val_min
        # pad binary segmentation map at the edges
        self.paddings_per_dim = [[(self.patch_size[dim_idx] + 1) // 2 - 1, (self.patch_size[dim_idx] + 1) // 2]
                                  for dim_idx in [1, 0]]
        # <<< diff: minimal uses padded_gt
        self.padded_img = F.pad(img, pad=flatten(self.paddings_per_dim),
                                mode='constant', value=self.padding_value) 
        self.min_patch_size = min(list(self.patch_size))

        # observation channel count = RGB + history + global information + brush_state
        self.observation_channel_count = 3 + self.history_size + 1 + 1    # <<< diff: minimal uses 1
        self.observation_shape = (*patch_size, self.observation_channel_count)
        self.observation_space = spaces.Box(low=min(-1, img_val_min), 
                                            high=max(img_val_max, 1),
                                            shape=self.observation_shape,
                                            dtype = np.float32)
        
        # define action space
        action_spaces = { # <<< diff: minimal only has delta_angle and brush_state
             'delta_angle': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'magnitude': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'brush_state': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'brush_radius': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'terminate': gym.spaces.Box(low=0, high=1, shape=(1,))
        }
        self.action_space = gym.spaces.Dict(action_spaces)      
        self.reset() # set default values

    def get_neutral_action(self):
        """Creates an initial neutral action, which is defined as 
        ('delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate')
        In the beginning of the agent, an initial state and observation is set by calling this function.
        
        Returns:
            observation: The first neutral observation given no prior agent action
        """        
        # <<< diff: minimal only returns [0.0, BRUSH_STATE_NOTHING]
        return torch.tensor([0.0, 0.0, BRUSH_STATE_NOTHING, 0.0, 0.0], dtype=torch.float, device=self.device)

    def reset(self):
        """Resets the environment state to its original state.
        """
        self.agent_pos = [int(dim_size // 2) for dim_size in self.img.shape[1:]]
        self.agent_angle = 0.0
        self.brush_width = 0  # <<< diff: minimal has no brush_width 
        self.brush_state = BRUSH_STATE_NOTHING
        self.history = torch.zeros((self.history_size, *self.patch_size), dtype=torch.float, device=self.device)
        
        self.seg_map_padded = F.pad(torch.zeros(self.img.shape[1:], device=self.device),
                                    pad=flatten(self.paddings_per_dim),
                                    mode='constant', value=self.padding_value)  

        self.padded_grid_x, self.padded_grid_y =\
            torch.meshgrid(torch.arange(self.seg_map_padded.shape[0], device=self.device),
                           torch.arange(self.seg_map_padded.shape[1], device=self.device),
                           indexing='xy')
        # train of though on dimensionality using meshgrid and indices:
        # patch_size = 5: [-1, -1, 0, -1, -1, -1]
        # patch_size = 4: [-1, 0, -1, -1]
        # comparison with below:
        # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
        # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]  (right-biased)
        
        self.padded_img_size = self.seg_map_padded.shape


        # <<< diff: minimal has no minimap
        self.minimap = torch.zeros(self.patch_size, dtype=torch.float, device=self.device) # global aggregation of past <history_size> predictions and current position
        # self.curr_pred = torch.zeros(patch_size, dtype=torch.float, device=self.device) # directly use seg_map
        self.terminated = False
        # Create a canvas to render the environment images upon 
        # self.canvas = np.ones(self.observation_shape) * 1
        self.seen_pixels = torch.zeros_like(self.img[0], dtype=torch.int8)

    def calculate_reward(self, delta_angle, new_brush_state, new_brush_radius, new_seen_pixels):
        """Given information on the latest action, calculates a (positive or negative) reward
        brush radius changes performed at the same time as angle changes are penalized less
        than brush size changes performed when the agent does not change the angle. This is to elevate the 
        underlying data structure, which is that a road usually has the same width. If the angle changes, 
        there might be a road crossing or a curve, which might impact the width of a road, which is why we
        allow this change more.
        However, this may lead to loopholes (agent turning right for a short time just to change the brush width,
        then turning )
        Args:
            delta_angle (float): the change of the brush angle
            new_brush_state (int): the brush state as a literal
            new_brush_radius (float): the new brush radius. The brush paints a circle, which is why radius is used
            new_seen_pixels (int): the amount of newly seen pixels due to the agents latest action

        Returns:
            reward, decomposed rewards, sum of decomposed rewards
        """
        
        reward_decomp_quantities = {k: 0 for k in self.rewards.keys()}
        reward_decomp_sums = {k: 0.0 for k in self.rewards.keys()}

        reward = 0.0
        
        normalize_tensor = lambda x: x.detach().cpu().numpy().item() if isinstance(x, torch.Tensor) else x

        if self.terminated:
            # get huge penalty for unseen pixels
            # also get penalty for false negatives? (NO)
            # new_seen_pixels are exclusive!
            num_unseen_pixels = self.img.shape[1]*self.img.shape[2] - self.seen_pixels.sum() - new_seen_pixels.sum()
            reward_delta = self.rewards['unseen_pix_pen'] * num_unseen_pixels
            reward -= reward_delta
            reward_decomp_quantities['unseen_pix_pen'] += normalize_tensor(num_unseen_pixels)
            reward_decomp_sums['unseen_pix_pen'] -= normalize_tensor(reward_delta)
            return reward, reward_decomp_quantities, reward_decomp_sums
        
        # changing brush state, angle and radius
        if self.brush_state != new_brush_state:
            reward -= self.rewards['changed_brush_pen']
            reward_decomp_quantities['changed_brush_pen'] += 1
            reward_decomp_sums['changed_brush_pen'] -= normalize_tensor(self.rewards['changed_brush_pen'])
        elif self.brush_state == BRUSH_STATE_PAINT:
            changed_angle_pen = torch.abs(delta_angle) * self.rewards['changed_angle_pen']
            reward -= changed_angle_pen
            reward_decomp_quantities['changed_angle_pen'] += 1
            reward_decomp_sums['changed_angle_pen'] -= normalize_tensor(changed_angle_pen)

            delta_brush_size = torch.abs(new_brush_radius-self.brush_width)
            changed_radius_pen = 1/(torch.max(torch.abs(delta_angle), torch.tensor(1e-1))) * delta_brush_size * self.rewards['changed_brush_rad_pen']
            reward -= changed_radius_pen
            reward_decomp_quantities['changed_brush_rad_pen'] += 1
            reward_decomp_sums['changed_brush_rad_pen'] -= normalize_tensor(changed_radius_pen)
        
        # sum errored prediction over complete segmentation state
        num_false_positives = torch.logical_and(new_seen_pixels, torch.logical_and(self.gt == 0, self.get_unpadded_segmentation() == 1)).sum()
        num_false_negatives = torch.logical_and(new_seen_pixels, torch.logical_and(self.gt == 1, self.get_unpadded_segmentation() == 0)).sum()
        
        false_pos_seg_pen = num_false_positives * self.rewards['false_pos_seg_pen']
        reward -= false_pos_seg_pen
        reward_decomp_quantities['false_pos_seg_pen'] += normalize_tensor(num_false_positives)
        reward_decomp_sums['false_pos_seg_pen'] -= normalize_tensor(false_pos_seg_pen)

        false_neg_seg_pen = num_false_negatives * self.rewards['false_neg_seg_pen']
        reward -= false_neg_seg_pen
        reward_decomp_quantities['false_neg_seg_pen'] += normalize_tensor(num_false_negatives)
        reward_decomp_sums['false_neg_seg_pen'] -= normalize_tensor(false_neg_seg_pen)
        
        # reward currently correctly predicted pixels
        num_new_true_pos = torch.logical_and(new_seen_pixels, torch.logical_and(self.gt == 1, self.get_unpadded_segmentation() == 1)).sum()
        num_new_true_neg = torch.logical_and(new_seen_pixels, torch.logical_and(self.gt == 0, self.get_unpadded_segmentation() == 0)).sum()
        
        true_pos_seg_rew = num_new_true_pos * self.rewards['true_pos_seg_rew']
        reward += true_pos_seg_rew
        reward_decomp_quantities['true_pos_seg_rew'] += normalize_tensor(num_new_true_pos)
        reward_decomp_sums['true_pos_seg_rew'] += normalize_tensor(true_pos_seg_rew)

        true_neg_seg_rew = num_new_true_neg * self.rewards['true_neg_seg_rew']
        reward += true_neg_seg_rew
        reward_decomp_quantities['true_neg_seg_rew'] += normalize_tensor(num_new_true_neg)
        reward_decomp_sums['true_neg_seg_rew'] += normalize_tensor(true_neg_seg_rew)
        
        # reward newly seen pixels
        num_newly_seen_pixels = torch.sum(new_seen_pixels)
        
        unseen_pix_rew = num_newly_seen_pixels * self.rewards['unseen_pix_rew']
        reward += unseen_pix_rew
        reward_decomp_quantities['unseen_pix_rew'] += normalize_tensor(num_newly_seen_pixels)
        reward_decomp_sums['unseen_pix_rew'] += normalize_tensor(unseen_pix_rew)
        
        return reward, reward_decomp_quantities, reward_decomp_sums

    def step(self, action):
        """process a new action, return the new observations, penalty and other information for the trainer
        Args:
            action (Tensor): 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'

        Returns:
            (new_observation, reward, done, new_info)
        """
        delta_angle, self.magnitude, new_brush_state, new_brush_radius, self.terminated = [action[idx] for idx in range(5)]
        # <<< diff: cannot erase in minimal!

        # calculate new segmentation
        # we don't need the old segmentation anymore, hence we do not store it
        if new_brush_state != BRUSH_STATE_NOTHING and new_brush_radius > 0:
            # implicit circle equation: (x-center_x)^2 + (y-center_y)^2 - r^2 = 0
            # inside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 < 0
            # outside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 > 0

            distance_map = (torch.square(self.padded_grid_y - (int(self.agent_pos[0]) + self.paddings_per_dim[0][0]))
                            + torch.square(self.padded_grid_x - (int(self.agent_pos[1]) + self.paddings_per_dim[1][0]) )
                            - new_brush_radius**2) # 2D
            unpadded_mask = self.seg_map_padded != self.padding_value
            stroke_ball = torch.logical_and(distance_map <= 0, unpadded_mask)  # boolean mask
            # paint: OR everything within stroke_ball with 1 (set to 1)
            # erase: AND everything within stroke_ball with 0 (set to 0)
            # current_seg, new_seg_map
            self.seg_map_padded[stroke_ball] = 1 if new_brush_state == BRUSH_STATE_PAINT else 0

            
        # calculate new_seen_pixels as we reward newly seen pixels to animate the agent to cover the whole segmentation map
        
        # unnormed_patch_coord_list: may exceed bounding box (be negative or >= width or >= height)
        unnormed_patch_coord_list = [[(int(dim_pos) - ((self.patch_size[dim_idx]+1)//2)) + 1, int(dim_pos) + ((self.patch_size[dim_idx]+2)//2)] for dim_idx, dim_pos in enumerate(self.agent_pos)]
        # reason for + 2:
        # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
        # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]  (right-biased)
        # assume + 1:
        # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
        # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1]  <--- too small!
        new_seen_pixels = torch.zeros_like(self.img[0], dtype=self.seen_pixels.dtype)
        new_seen_pixels[max(0, unnormed_patch_coord_list[0][0]) : min(self.seen_pixels.shape[0], unnormed_patch_coord_list[0][1]),
                        max(0, unnormed_patch_coord_list[1][0]) : min(self.seen_pixels.shape[0], unnormed_patch_coord_list[1][1])] = 1
        new_seen_pixels = torch.max(new_seen_pixels - self.seen_pixels, torch.zeros_like(new_seen_pixels, dtype=new_seen_pixels.dtype)) > 0
        
        # calculate reward
        reward, reward_decomp_quantities, reward_decomp_sums =\
            self.calculate_reward(delta_angle, new_brush_state, new_brush_radius, new_seen_pixels)
        
        # update class values
        # current position is marked separately, and mark is added right before the minimap is forwarded to
        def get_minimap_pixel_coords(original_coords):
            return tuple([int(pos // (self.img.shape[dim_idx+1] / self.patch_size[dim_idx])) for dim_idx, pos in enumerate(original_coords)])
        self.minimap[get_minimap_pixel_coords(self.agent_pos)] = 1
        # calculate new position
        self.angle = self.agent_angle + delta_angle * math.pi
        delta_x = math.cos(self.angle) * self.magnitude
        delta_y = math.sin(self.angle) * self.magnitude
        # project to bounding box
        # do not round here, since if the changes are too small, the position may never get updated
        self.agent_pos = [max(0, min(self.agent_pos[0] + delta_y, self.img_size[0] - 1)),
                          max(0, min(self.agent_pos[1] + delta_x, self.img_size[1] - 1))]
        self.brush_state = new_brush_state
        self.brush_width = new_brush_radius
        self.seen_pixels[new_seen_pixels] = 1
        
        # extract padded patch, but shift indices so that they start from 0 (instead of possibly negative indices due to padding)
        start_dim_0 = self.paddings_per_dim[0][0] + unnormed_patch_coord_list[0][0]
        end_dim_0 = self.paddings_per_dim[0][0] + unnormed_patch_coord_list[0][1]
        start_dim_1 = self.paddings_per_dim[1][0] + unnormed_patch_coord_list[1][0]
        end_dim_1 = self.paddings_per_dim[1][0] + unnormed_patch_coord_list[1][1]
        new_padded_patch = self.seg_map_padded[start_dim_0:end_dim_0, start_dim_1:end_dim_1]
        new_rgb_patch = self.padded_img[:, start_dim_0:end_dim_0, start_dim_1:end_dim_1]

        # seg_map is padded; add padding size to patch_coords (to correct negative offset)
        self.history = torch.cat((self.history[1:], new_padded_patch.unsqueeze(0)), dim=0)

        global_information = self.minimap.clone()

        global_information[get_minimap_pixel_coords(self.agent_pos)] = 0.5
        brush_state = torch.ones_like(self.minimap) * self.brush_state
        new_observation = torch.cat([new_rgb_patch, self.history, self.minimap.unsqueeze(0), brush_state.unsqueeze(0)], dim=0)
        new_info = {'reward_decomp_quantities': reward_decomp_quantities,
                    'reward_decomp_sums': reward_decomp_sums}
        return new_observation, reward, self.terminated, new_info
    
    def get_unpadded_segmentation(self):
        """ Retrieves the unpadded segmentation, so that the trainer can calculate the loss easily during testing
        """
        paddings_dim_0, paddings_dim_1 = self.paddings_per_dim
        i, j = paddings_dim_0
        k, l = paddings_dim_1
        return self.seg_map_padded[i:-j, k:-l]

    def get_rounded_agent_pos(self):
        """returns the rounded agent pos
        """
        return [int(self.agent_pos[0]), int(self.agent_pos[1])]


class SegmentationEnvironmentMinimal(Env):
    """Minimal Environment for Reinforcement Learning Approach
    It calculates the next inputs for the model based on the models output, as well as calculates the rewards
    and penalties. In general it saves the state of the agent and environment. This minimal version uses a less complex
    reward/penalty system as well as less information as the model's input and a smaller action space
    Args:
        gt (torch.Tensor): ground-truth of image to segment, or None if no segmentation is to be performed
        img_val_min (int or float): minimum value of a pixel in input image
        img_val_max (int or float): maximum value of a pixel in input image
        is_supervised: indicates whether to use a supervised environment based on an automatically computed off-policy exploration policy
                        if True, the policy network is assumed to output "angle", "brush_radius" and ; otherwise, network is 
        patch_size (tuple of int): size of patch visible to agent at a given timestep
        rewards (dictionary): a dictionary of rewards, penalties have a negative sign
        supervision_optimal_brush_radius_map_paths (list of str): paths to pickled files with map of optimal radii for each pixel, or None if is_supervised is False
        supervision_non_max_suppressed_map (list of str): map of values after performing non-maximum suppression on optimal brush radius maps, or None if is_supervised is False
        exploration_model_action_ratio (float): the ratio of actions to sample from the model for the exploration policy
    """
    def __init__(self, img_paths, gt_paths, patch_size, img_val_min, img_val_max, is_supervised=False, rewards=DEFAULT_REWARDS_MINIMAL,
                 supervision_optimal_brush_radius_map_paths=None,
                 supervision_non_max_suppressed_map_paths=None,
                 sample_preprocessing=None,
                 dl_sample_idxs=None,
                 exploration_model_action_ratio=1.0):
        
        super(SegmentationEnvironmentMinimal, self).__init__()
        print('SegmentationEnvironmentMinimal __init__ entered')

        self.sample_idx = -1
        self.patch_size = patch_size
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.rewards = rewards
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_val_min = img_val_min
        self.img_val_max = img_val_max
        self.padding_value = -1 if img_val_min == 0 else -2 * img_val_min
        # pad binary segmentation map at the edges
        self.paddings_per_dim = [[(self.patch_size[dim_idx] + 1) // 2 - 1, (self.patch_size[dim_idx] + 1) // 2]
                                  for dim_idx in [1, 0]]

        # Groundtruth is Input
        self.observation_channel_count = 1
        self.observation_shape = (*patch_size, self.observation_channel_count)
        self.observation_space = spaces.Box(low=0, 
                                            high=1,
                                            shape=self.observation_shape,
                                            dtype = np.float32)
        
        # delta_angle, magnitude brush_state, brush_radius, terminate?
        self.min_patch_size = min(list(self.patch_size))

        self.magnitude = 5.0 if is_supervised else 1.0  # 1.0 # 1/self.min_patch_size # magnitude is always one pixel per step (if in unsupervised setting)
        self.brush_width = 1 # paint one pixel per step (if in unsupervised setting)
        self.is_supervised = is_supervised
        self.sample_preprocessing = sample_preprocessing
        self.dl_sample_idxs = dl_sample_idxs
        
        if is_supervised:
            action_spaces = {
                'angle': gym.spaces.Box(low=0, high=1, shape=(1,)),
                'brush_radius': gym.spaces.Box(low=0, high=1, shape=(1,)),
                'magnitude': gym.spaces.Box(low=0, high=1, shape=(1,))
            }
        else:
            action_spaces = {
                'delta_angle': gym.spaces.Box(low=0, high=1, shape=(1,)),
                'brush_state': gym.spaces.Box(low=0, high=1, shape=(1,))
            }
        self.action_space = gym.spaces.Dict(action_spaces)
                                                  
        self.supervision_optimal_brush_radius_map_paths = supervision_optimal_brush_radius_map_paths
        self.supervision_non_max_suppressed_map_paths = supervision_non_max_suppressed_map_paths

        self.exploration_model_action_ratio = exploration_model_action_ratio

        print('SegmentationEnvironmentMinimal __init__: about to call reset()')

        self.reset()
        
        print('SegmentationEnvironmentMinimal __init__ left')

    @staticmethod
    def get_neutral_action():
        # returns a neutral action
        # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'
        
        # <<< diff: minimal only returns [0.0, BRUSH_STATE_NOTHING]
        return torch.tensor([0.0, 0.0, BRUSH_STATE_NOTHING, 0.0, 0.0], dtype=torch.float,
                            device='cuda' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        # move to next sample
        self.sample_idx = (self.sample_idx + 1) % len(self.img_paths)

        img_path = self.img_paths[self.sample_idx]

        # load the image from disk
        img_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.img = torch.from_numpy(img_np)
        self.img = torch.permute(self.img, (2, 0, 1))  # channel dim first
        # BGR to RGB
        self.img = self.img[[2, 1, 0, 3] if self.img.shape[0] == 4 else [2, 1, 0] if self.img.shape[0] == 3 else [0]]
        self.img = self.img.to(self.device)
        
        if self.sample_preprocessing is not None:
            self.img = self.sample_preprocessing(self.img, is_gt=False)

        if self.gt_paths is not None:
            gt_path = self.gt_paths[self.sample_idx]
            # there is groundtruth
            gt_np = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            self.gt = torch.from_numpy(gt_np)
            if len(self.gt.shape) == 3:
                self.gt = torch.permute(self.gt, (2, 0, 1))  # channel dim first
                # BGR to RGB
                self.gt = self.gt[[2, 1, 0, 3] if self.gt.shape[0] == 4 else [2, 1, 0] if self.gt.shape[0] == 3 else [0]]
            else:
                self.gt = self.gt.unsqueeze(0)
            
            if self.sample_preprocessing is not None:
                self.gt = self.sample_preprocessing(self.gt, is_gt=True)


            while len(self.gt.shape) >= 3:
                self.gt = self.gt[0]
            
            self.gt = self.gt.to(self.device)
            self.padded_gt = F.pad(self.gt, pad=flatten(self.paddings_per_dim),
                                    mode='constant', value=self.padding_value)
        else:
            self.gt = None
            self.padded_gt = None
        
        if self.supervision_optimal_brush_radius_map_paths is not None:
            with open(self.supervision_optimal_brush_radius_map_paths[self.sample_idx], 'rb') as f:
                self.supervision_optimal_brush_radius_map = torch.from_numpy(pickle.load(f))
        else:
            self.supervision_optimal_brush_radius_map = None

        if self.supervision_non_max_suppressed_map_paths is not None:
            with open(self.supervision_non_max_suppressed_map_paths[self.sample_idx], 'rb') as f:
                self.supervision_non_max_suppressed_map = torch.from_numpy(pickle.load(f))
        else:
            self.supervision_non_max_suppressed_map = None

        self.img_size = self.img.shape[1:]

        self.padded_img = F.pad(self.img, pad=flatten(self.paddings_per_dim),
                                mode='constant', value=self.padding_value)

        self.agent_pos = [int(dim_size // 2) for dim_size in self.img_size]
        self.agent_angle = 0.0
        self.brush_width = 0.0 if self.is_supervised else 1.0
        self.brush_state = BRUSH_STATE_NOTHING
        self.seg_map_padded = F.pad(torch.zeros(self.img_size, device=self.device),
                                    pad=flatten(self.paddings_per_dim),
                                    mode='constant', value=self.padding_value)  

        self.padded_grid_x, self.padded_grid_y =\
            torch.meshgrid(torch.arange(self.seg_map_padded.shape[0], device=self.device),
                           torch.arange(self.seg_map_padded.shape[1], device=self.device),
                           indexing='xy')

        self.grid_x, self.grid_y =\
            torch.meshgrid(torch.arange(self.img_size[0], device=self.device),
                           torch.arange(self.img_size[1], device=self.device),
                           indexing='xy')
        
        self.seen_pixels = torch.zeros(self.img_size, dtype=torch.int8, device=self.img.device)
        
        # [0]
        # patch_size=5: [-1, -1, 0, -1, -1, -1]
        # patch_size=4: [-1, 0, -1, -1]

        # comparison with below:

        # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
        # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]  (right-biased)
        
        self.padded_img_size = self.seg_map_padded.shape
        
        self.minimap = torch.zeros(self.patch_size, dtype=torch.float, device=self.device) # global aggregation of past <history_size> predictions and current position

        if self.is_supervised and self.gt is not None:
            # initialization of exploration policy
            # (as in walker.py)

            dims = self.img_size
            y, x = float(random.randint(0, dims[0]-1)), float(random.randint(0, dims[1]-1))

            last_y_int, last_x_int = int(np.round(y)), int(np.round(x))
            
            is_on_road = self.gt[last_y_int, last_x_int] > 0
            is_on_max = self.supervision_non_max_suppressed_map[last_y_int, last_x_int] > 0
            
            self.policy_state = {'is_on_road': is_on_road,
                                 'is_on_max': is_on_max,
                                 'last_angle': 0.0,
                                 'last_x_int': last_x_int,
                                 'last_y_int': last_y_int,
                                 'current_is_visited': False,
                                 'visited': torch.zeros(dims, device=self.img.device, dtype=torch.int8)}
        else:
            self.policy_state = None

        self.info_sum = {'reward_decomp_quantities': {}, 'reward_decomp_sums': {}}
        return torch.tensor(0)  # must return a Tensor

    def calculate_reward(self, stroke_ball, new_seen_pixels):
        # we penalize "brush radius changes performed at the same time as angle changes" less
        # than "brush size changes performed when the agent does not "$
        # this may lead to loopholes (agent turning right for a short time just to change the brush width,
        # then turning )
        
        reward_decomp_quantities = {k: 0 for k in self.rewards.keys()}
        reward_decomp_sums = {k: 0.0 for k in self.rewards.keys()}

        reward = 0.0
        
        normalize_tensor = lambda x: x.detach().cpu().numpy().item() if isinstance(x, torch.Tensor) else x
        new_seg = self.get_unpadded_segmentation()[stroke_ball]
        
        # reward newly seen pixels
        num_newly_seen_pixels = torch.sum(new_seen_pixels)
        
        unseen_pix_rew = num_newly_seen_pixels * self.rewards['unseen_pix_rew']
        reward += unseen_pix_rew
        reward_decomp_quantities['unseen_pix_rew'] += normalize_tensor(num_newly_seen_pixels)
        reward_decomp_sums['unseen_pix_rew'] += normalize_tensor(unseen_pix_rew)
        
        # reward currently correctly predicted pixels
        num_new_true_pos = torch.logical_and(self.gt[stroke_ball] == 1, new_seg == 1).sum()
        num_new_true_neg = torch.logical_and(self.gt[stroke_ball] == 0, new_seg == 0).sum()
        
        true_pos_seg_rew = num_new_true_pos * self.rewards['true_seg_rew']
        reward += true_pos_seg_rew
        reward_decomp_quantities['true_seg_rew'] += normalize_tensor(num_new_true_pos)
        reward_decomp_sums['true_seg_rew'] += normalize_tensor(true_pos_seg_rew)

        true_neg_seg_rew = num_new_true_neg * self.rewards['true_seg_rew']
        reward += true_neg_seg_rew
        reward_decomp_quantities['true_seg_rew'] += normalize_tensor(num_new_true_neg)
        reward_decomp_sums['true_seg_rew'] += normalize_tensor(true_neg_seg_rew)
        
        # sum errored prediction over the current painted / unpainted pixel
        num_false_positives = torch.logical_and(self.gt[stroke_ball] == 0, new_seg == 1).sum()
        num_false_negatives = torch.logical_and(self.gt[stroke_ball] == 1, new_seg == 0).sum()
        
        false_pos_seg_pen = num_false_positives * self.rewards['true_seg_rew'] * 0.21651089 # true pos reward constant * (road_pixels/non-road_pixels)
        reward -= false_pos_seg_pen
        reward_decomp_quantities['true_seg_rew'] += normalize_tensor(num_false_positives)
        reward_decomp_sums['true_seg_rew'] -= normalize_tensor(false_pos_seg_pen)

        false_neg_seg_pen = num_false_negatives * self.rewards['true_seg_rew'] * 0.21651089
        reward -= false_neg_seg_pen
        reward_decomp_quantities['true_seg_rew'] += normalize_tensor(num_false_negatives)
        reward_decomp_sums['true_seg_rew'] -= normalize_tensor(false_neg_seg_pen)
        
        return reward, reward_decomp_quantities, reward_decomp_sums

    def step(self, action):
        # action has already been unnormalized!

        # returns: (new_observation, reward, done, new_info)
        # action is Tensor, with highest dimension containing: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'

        def get_minimap_pixel_coords(original_coords):
            return tuple([int(pos // (self.img.shape[dim_idx+1] / self.patch_size[dim_idx])) for dim_idx, pos in enumerate(original_coords)])
        
        terminated = False
        supervision_desired_outputs = None

        # here, we define the exploration policy
        if self.is_supervised:
            # distinguish whether GT is available or not
            # if GT available, use "supervised" exploration policy
            # else, use model's outputs

            if None not in [self.supervision_optimal_brush_radius_map, self.supervision_non_max_suppressed_map]:
                LARGE_VALUE = 2**30  # needed because "distance_map_no_brush" has dtype long; cannot use inf

                distance_map_no_brush = torch.square(self.grid_y - self.agent_pos[0]) + torch.square(self.grid_x - self.agent_pos[1])
                distance_map_no_brush[self.supervision_non_max_suppressed_map == 0] = LARGE_VALUE
                distance_map_no_brush[self.policy_state['visited'] > 0] = LARGE_VALUE
                
                if distance_map_no_brush.min() == LARGE_VALUE:
                    terminated = True
                else:
                    closest_ys, closest_xs = torch.where(distance_map_no_brush == distance_map_no_brush.min())
                        
                    lowest_delta_angle = torch.tensor(torch.nan)
                    lowest_delta_magnitude = torch.tensor(torch.nan)

                    final_closest_y, final_closest_x = None, None

                    for closest_y, closest_x in zip(closest_ys, closest_xs):
                        tmp_angle = torch.arctan2(closest_y - self.agent_pos[0], closest_x - self.agent_pos[1])
                        if torch.isnan(lowest_delta_angle) or torch.abs(tmp_angle - self.policy_state['last_angle']) < torch.abs(lowest_delta_angle - self.policy_state['last_angle']):
                            lowest_delta_angle = tmp_angle
                            lowest_delta_magnitude = torch.sqrt( (closest_y - self.agent_pos[0])**2.0 + (closest_x - self.agent_pos[1])**2.0 ).item()
                            final_closest_y = closest_y
                            final_closest_x = closest_x

                    magnitude = min(self.magnitude, lowest_delta_magnitude)
                    new_brush_radius = self.supervision_optimal_brush_radius_map[self.policy_state['last_y_int'], self.policy_state['last_x_int']]
                    
                    # note: delta in lowest _delta_ angle refers to delta from policy_state['last_angle']
                    delta_angle = lowest_delta_angle - self.policy_state['last_angle']

                    # the loss is minimized between the unnormalized model outputs and the desired outputs we just
                    # calculated
                    supervision_desired_outputs = torch.tensor([lowest_delta_angle, new_brush_radius, magnitude],
                                                                device=self.img.device, dtype=self.img.dtype)

                    if np.random.uniform() <= self.exploration_model_action_ratio:
                        # still use model's output
                        angle, new_brush_radius, magnitude = [action[idx] for idx in range(3)]
                    
                new_brush_state = BRUSH_STATE_PAINT
            else:
                angle, new_brush_radius, magnitude = [action[idx] for idx in range(3)]
                delta_angle = angle - self.agent_angle
                new_brush_state = BRUSH_STATE_PAINT
        else:
            delta_angle, new_brush_state = [action[idx] for idx in range(2)]
            magnitude = self.magnitude
            new_brush_radius = self.brush_width
            
        unnormed_patch_coord_list = [[(int(dim_pos) - ((self.patch_size[dim_idx]+1)//2)) + 1, int(dim_pos) + ((self.patch_size[dim_idx]+2)//2)] for dim_idx, dim_pos in enumerate(self.agent_pos)]
        reward, reward_decomp_quantities, reward_decomp_sums = torch.tensor(0.0), {}, {}

        if not terminated:
            # calculate new segmentation
            # we don't need the old segmentation anymore, hence we do not store it
            
            # implicit circle equation: (x-center_x)^2 + (y-center_y)^2 - r^2 = 0
            # inside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 < 0
            # outside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 > 0

            # x: [0, 1, 2, 3, ...] (row vector)
            # y: [0, 1, 2, 3, ...] (column vector)

            if torch.round(new_brush_radius) > 0:
                distance_map = (torch.square(self.padded_grid_y - (int(self.agent_pos[0]) + self.paddings_per_dim[0][0]))
                                + torch.square(self.padded_grid_x - (int(self.agent_pos[1]) + self.paddings_per_dim[0][1]) )
                                - new_brush_radius**2) # 2D
                unpadded_mask = self.seg_map_padded != self.padding_value
                stroke_ball = torch.logical_and(distance_map <= 0, unpadded_mask)  # boolean mask
                distance_map_unpadded = (torch.square(self.grid_y - max(0,min(len(self.grid_y)-1, int(self.agent_pos[0]))))
                                        + torch.square(self.grid_x - max(0,min(len(self.grid_x)-1, int(self.agent_pos[1]))))
                                        - new_brush_radius**2) # 2D
                stroke_ball_unpadded = distance_map_unpadded <= 0
                # paint: OR everything within stroke_ball with 1 (set to 1)
                # erase: AND everything within stroke_ball with 0 (set to 0)
                #current_seg, new_seg_map
                self.seg_map_padded[stroke_ball] = 1 if new_brush_state == BRUSH_STATE_PAINT else 0

                if self.is_supervised and None not in [self.supervision_optimal_brush_radius_map, self.supervision_non_max_suppressed_map]:
                    if self.policy_state['is_on_road']:
                        self.policy_state['visited'][stroke_ball_unpadded] = 1

            
            # unnormed_patch_coord_list: may exceed bounding box (be negative or >= width or >= height)
            # reason for + 2:
            # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
            # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]  (right-biased)
            
            # assume + 1:
            # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
            # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1]  <--- too small!
            
            new_seen_pixels = torch.zeros_like(self.img[0], dtype=self.seen_pixels.dtype)
            # new_pixel_idx = tuple([[max(0, dim_range[0]), min(self.seen_pixels.shape[dim_idx], dim_range[1])] for dim_idx, dim_range in enumerate(unnormed_patch_coord_list)])
            new_seen_pixels[max(0, unnormed_patch_coord_list[0][0]) : min(self.seen_pixels.shape[0], unnormed_patch_coord_list[0][1]),
                            max(0, unnormed_patch_coord_list[1][0]) : min(self.seen_pixels.shape[0], unnormed_patch_coord_list[1][1])] = 1
            new_seen_pixels = torch.max(new_seen_pixels - self.seen_pixels, torch.zeros_like(new_seen_pixels, dtype=new_seen_pixels.dtype)) > 0
            
            # calculate reward
            if not self.is_supervised:
                reward, reward_decomp_quantities, reward_decomp_sums =\
                    self.calculate_reward(stroke_ball_unpadded, new_seen_pixels)

            self.minimap[get_minimap_pixel_coords(self.agent_pos)] = 1

            # update class values

            # calculate new position
            self.agent_angle = torch.clip(self.agent_angle + delta_angle, -torch.pi, torch.pi)
            delta_x = math.cos(self.agent_angle) * magnitude
            delta_y = math.sin(self.agent_angle) * magnitude
            # project to bounding box
            # do not round here, since if the changes are too small, the position may never get updated
            self.agent_pos = [max(0, min(self.agent_pos[0] + delta_y, self.img_size[0] - 1)),
                              max(0, min(self.agent_pos[1] + delta_x, self.img_size[1] - 1))]
            self.brush_state = new_brush_state
            
            self.seen_pixels[new_seen_pixels] = 1
            
            if self.is_supervised and None not in [self.supervision_optimal_brush_radius_map, self.supervision_non_max_suppressed_map]:
                new_y_int, new_x_int = self.get_rounded_agent_pos()

                self.policy_state['is_on_road'] = self.supervision_optimal_brush_radius_map[new_y_int, new_x_int] > 0
                self.policy_state['is_on_max'] = self.supervision_non_max_suppressed_map[new_y_int, new_x_int] > 0

                self.policy_state['current_is_visited'] = self.policy_state['visited'][new_y_int, new_x_int] > 0

                for fill_y in range(min(self.policy_state['last_y_int'], new_y_int), max(self.policy_state['last_y_int'], new_y_int)):
                    for fill_x in range(min(self.policy_state['last_x_int'], new_x_int), max(self.policy_state['last_x_int'], new_x_int)):
                        self.policy_state['visited'][fill_y, fill_x] = 1

                self.policy_state['last_y_int'], self.policy_state['last_x_int'] = new_y_int, new_x_int
                self.policy_state['last_angle'] = lowest_delta_angle

        # extract padded patch, but shift indices so that they start from 0 (instead of possibly negative indices due to padding)
        start_dim_0 = self.paddings_per_dim[0][0] + unnormed_patch_coord_list[0][0]
        end_dim_0 = self.paddings_per_dim[0][0] + unnormed_patch_coord_list[0][1]
        start_dim_1 = self.paddings_per_dim[1][0] + unnormed_patch_coord_list[1][0]
        end_dim_1 = self.paddings_per_dim[1][0] + unnormed_patch_coord_list[1][1]
        # new_padded_patch = self.seg_map_padded[start_dim_0:end_dim_0, start_dim_1:end_dim_1]
        new_rgb_patch = self.padded_img[:, start_dim_0:end_dim_0, start_dim_1:end_dim_1]

        global_information = self.minimap.clone()

        global_information[get_minimap_pixel_coords(self.agent_pos)] = 0.5

        # also append a map of the currently segmented areas

        # F.interpolate expects B, C, H, W --> unsqueeze self.get_unpadded_segmentation() twice: B=1, C=1
        # then remove batch dim from result
        red_seg =\
            F.interpolate(self.get_unpadded_segmentation().unsqueeze(0).unsqueeze(0), self.minimap.shape).squeeze(0)

        new_observation = torch.cat([new_rgb_patch, self.minimap.unsqueeze(0), red_seg], dim=0)
        new_info = {'reward_decomp_quantities': reward_decomp_quantities,
                    'reward_decomp_sums': reward_decomp_sums}
        
        if supervision_desired_outputs is not None:
            new_info['supervision_desired_outputs'] = supervision_desired_outputs
        
        new_info['unpadded_segmentation'] = self.get_unpadded_segmentation().float().detach()
        new_info['rounded_agent_pos'] = self.get_rounded_agent_pos()


        self.info_sum['reward_decomp_quantities'] = {k: self.info_sum['reward_decomp_quantities'].get(k, 0.0) + v for k, v in reward_decomp_quantities.items()}
        self.info_sum['reward_decomp_sums'] = {k: self.info_sum['reward_decomp_sums'].get(k, 0.0) + v for k, v in reward_decomp_sums.items()}

        new_info['info_sum'] = self.info_sum
        if self.dl_sample_idxs is not None:
            new_info['sample_idx'] = self.dl_sample_idxs[self.sample_idx]
        new_info['sample_sequence_idx'] = self.sample_idx
        return new_observation, reward, torch.tensor(terminated, device=self.img.device), new_info
    
    def get_unpadded_segmentation(self): # for trainer to calculate loss during testing
        paddings_dim_0, paddings_dim_1 = self.paddings_per_dim
        i, j = paddings_dim_0
        k, l = paddings_dim_1
        return torch.clone(self.seg_map_padded[i:-j, k:-l])

    def get_rounded_agent_pos(self):
        return [int(self.agent_pos[0]), int(self.agent_pos[1])]
