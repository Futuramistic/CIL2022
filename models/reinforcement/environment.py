import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
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


class SegmentationEnvironment(Env):

    def __init__(self, img, gt, patch_size, history_size, img_val_min, img_val_max, rewards=DEFAULT_REWARDS):
        """Environment for reinforcement learning

        Args:
            img (torch.Tensor): image to segment, dimension (img_size, img_channels)
            gt (torch.Tensor): ground-truth of image to segment, or None if no segmentation is to be performed
            patch_size (tuple of int): size of patch visible to agent at a given timestep
            history_size (int): how many previous predictions of the network should be fed into the network
            img_val_min (int or float): minimum value of a pixel in input image
            img_val_max (int or float): maximum value of a pixel in input image
        """
        super(SegmentationEnvironment, self).__init__()
        self.patch_size = patch_size
        self.img = img
        self.gt = gt
        self.img_size = img.shape[1:]
        self.rewards = rewards
        self.device = self.img.device
        # self.padded_img_size = [dim_size + torch.ceil(dim_size / 2) * 2 for dim_size in img.shape]
        self.history_size = history_size
        self.img_val_min = img_val_min
        self.img_val_max = img_val_max
        self.padding_value = -1 if img_val_min == 0 else -2 * img_val_min
        # pad binary segmentation map at the edges
        self.paddings_per_dim = [[(self.patch_size[dim_idx] + 1) // 2 - 1, (self.patch_size[dim_idx] + 1) // 2]
                                  for dim_idx in [1, 0]]

        # RGB + history + global information + brush_state
        self.observation_channel_count = 3 + self.history_size + 1 + 1
        self.observation_shape = (*patch_size, self.observation_channel_count)
        self.observation_space = spaces.Box(low=min(-1, img_val_min), 
                                            high=max(img_val_max, 1),
                                            shape=self.observation_shape,
                                            dtype = np.float32)
        
        # delta_angle, magnitude brush_state, brush_radius, terminate?
        self.min_patch_size = min(list(self.patch_size))
        # action_spaces = {
        #     'delta_angle': gym.spaces.Box(low=0, high=1, shape=(1,)),
        #     'magnitude': gym.spaces.Box(low=0, high=self.min_patch_size / 2, shape=(1,)),
        #     'brush_state': gym.spaces.Discrete(3, start=-1), # {-1, 0, 1} = {BRUSH_STATE_ERASE, BRUSH_STATE_NOTHING, BRUSH_STATE_PAINT}
        #     'brush_radius': gym.spaces.Box(low=0, high=self.min_patch_size / 2, shape=(1,)),
        #     'terminate': gym.spaces.Discrete(2) # {0, 1} = {TERMINATE_NO, TERMINATE_YES}
        # }

        action_spaces = {
             'delta_angle': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'magnitude': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'brush_state': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'brush_radius': gym.spaces.Box(low=0, high=1, shape=(1,)),
             'terminate': gym.spaces.Box(low=0, high=1, shape=(1,))
        }
        self.action_space = gym.spaces.Dict(action_spaces)
                                                  
        self.padded_img = F.pad(img, pad=flatten(self.paddings_per_dim),
                                mode='constant', value=self.padding_value) 
    
        self.reset()        

    def get_neutral_action(self):
        # returns a neutral action
        # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'
        return torch.tensor([0.0, 0.0, BRUSH_STATE_NOTHING, 0.0, 0.0], dtype=torch.float, device=self.device)

    def reset(self):
        self.agent_pos = [int(dim_size // 2) for dim_size in self.img.shape[1:]]
        self.agent_angle = 0.0
        self.brush_width = 0
        self.brush_state = BRUSH_STATE_NOTHING
        self.history = torch.zeros((self.history_size, *self.patch_size), dtype=torch.float, device=self.device)
        
        self.seg_map_padded = F.pad(torch.zeros(self.img.shape[1:], device=self.device),
                                    pad=flatten(self.paddings_per_dim),
                                    mode='constant', value=self.padding_value)  

        self.padded_grid_x, self.padded_grid_y =\
            torch.meshgrid(torch.arange(self.seg_map_padded.shape[0], device=self.device),
                           torch.arange(self.seg_map_padded.shape[1], device=self.device),
                           indexing='xy')
        
        # [0]
        # patch_size=5: [-1, -1, 0, -1, -1, -1]
        # patch_size=4: [-1, 0, -1, -1]

        # comparison with below:

        # patch_size = 5 (odd) : [dim_size - 2, dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]
        # patch_size = 4 (even): [dim_size - 1, dim_size + 0, dim_size + 1, dim_size + 2]  (right-biased)
        
        self.padded_img_size = self.seg_map_padded.shape
        self.minimap = torch.zeros(self.patch_size, dtype=torch.float, device=self.device) # global aggregation of past <history_size> predictions and current position
        # self.curr_pred = torch.zeros(patch_size, dtype=torch.float, device=self.device) # directly use seg_map
        self.terminated = False
        # Create a canvas to render the environment images upon 
        # self.canvas = np.ones(self.observation_shape) * 1
        self.seen_pixels = torch.zeros_like(self.img[0], dtype=torch.int8)

    def calculate_reward(self, delta_angle, new_brush_state, new_brush_radius,
                         new_seen_pixels):
        # we penalize "brush radius changes performed at the same time as angle changes" less
        # than "brush size changes performed when the agent does not "$
        # this may lead to loopholes (agent turning right for a short time just to change the brush width,
        # then turning )
        
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
        reward_decomp_sums['false_pos_seg_pen'] -= normalize_tensor(false_neg_seg_pen)
        
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
        # returns: (new_observation, reward, done, new_info)
        # action is Tensor, with highest dimension containing: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'
        # unpack
        delta_angle, self.magnitude, new_brush_state, new_brush_radius, self.terminated = [action[idx] for idx in range(5)]
        
        # calculate new segmentation
        # we don't need the old segmentation anymore, hence we do not store it
        if new_brush_state != BRUSH_STATE_NOTHING and new_brush_radius > 0:
            # implicit circle equation: (x-center_x)^2 + (y-center_y)^2 - r^2 = 0
            # inside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 < 0
            # outside circle: (x-center_x)^2 + (y-center_y)^2 - r^2 > 0

            # x: [0, 1, 2, 3, ...] (row vector)
            # y: [0, 1, 2, 3, ...] (column vector)

            distance_map = (torch.square(self.padded_grid_y - (self.agent_pos[0] + self.paddings_per_dim[0][0]))
                            + torch.square(self.padded_grid_x - (self.agent_pos[1] + self.paddings_per_dim[1][0]) )
                            - new_brush_radius**2) # 2D
            unpadded_mask = self.seg_map_padded != self.padding_value
            stroke_ball = torch.logical_and(distance_map <= 0, unpadded_mask)  # boolean mask
            # paint: OR everything within stroke_ball with 1 (set to 1)
            # erase: AND everything within stroke_ball with 0 (set to 0)
            #current_seg, new_seg_map
            self.seg_map_padded[stroke_ball] = 1 if new_brush_state == BRUSH_STATE_PAINT else 0

            
        # calculate new_seen_pixels
        
        # unnormed_patch_coord_list: may exceed bounding box (be negative or >= width or >= height)
        unnormed_patch_coord_list = [[(dim_pos - ((self.patch_size[dim_idx]+1)//2)) + 1, dim_pos + ((self.patch_size[dim_idx]+2)//2)] for dim_idx, dim_pos in enumerate(self.agent_pos)]
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
        self.agent_pos = [max(0, min(int(self.agent_pos[0] + delta_y), self.img_size[0] - 1)),
                          max(0, min(int(self.agent_pos[1] + delta_x), self.img_size[1] - 1))]
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
    
    def get_unpadded_segmentation(self): # for trainer to calculate loss during testing
        paddings_dim_0, paddings_dim_1 = self.paddings_per_dim
        i, j = paddings_dim_0
        k, l = paddings_dim_1
        return self.seg_map_padded[i:-j, k:-l]
