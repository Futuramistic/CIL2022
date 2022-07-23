# part of code adapted from https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py

import argparse
import cv2
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import pickle

from PIL import Image
from tqdm import tqdm

"""
Compute and pickle the pairwise distances between samples of two datasets
"""


class PartialInceptionNetwork(nn.Module):
    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        if x.shape[1:] != (3, 299, 299):
            x = F.interpolate(x, (299, 299), mode='bilinear')

        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
                                             ", but got {}".format(x.shape)
        x = x * 2 - 1  # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = F.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def main(dataset_1, dataset_2):
    dist_dict_1_2 = {}
    dist_dict_2_1 = {}

    ds_1_images_dir = os.path.join('dataset', dataset_1, 'training', 'images')
    if not os.path.isdir(ds_1_images_dir):
        print(f'Directory "{ds_1_images_dir}" (supposed to contain images from which to compute '
              f'dataset stats) does not exist')
        return

    ds_2_images_dir = os.path.join('dataset', dataset_2, 'training', 'images')
    if not os.path.isdir(ds_2_images_dir):
        print(f'Directory "{ds_2_images_dir}" (supposed to contain images from which to compute '
              f'dataset stats) does not exist')
        return

    # load model

    model = PartialInceptionNetwork()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    for filename_1 in tqdm(os.listdir(ds_1_images_dir)):
        if filename_1.lower().endswith('.png'):
            # load the image from disk
            img_path_1 = os.path.join(ds_1_images_dir, filename_1)
            image_1_np = cv2.imread(img_path_1, cv2.IMREAD_UNCHANGED)
            image_1 = torch.from_numpy(image_1_np)
            if len(image_1.shape) == 2:
                image_1 = image_1.unsqueeze(-1).repeat((1, 1, 3))
            image_1 = torch.permute(image_1, (2, 0, 1))  # channel dim first
            image_1 = image_1[
                [2, 1, 0, 3] if image_1.shape[0] == 4 else [2, 1, 0] if image_1.shape[0] == 3 else [0]]  # BGR to RGB
            image_1 = image_1[:3, ...].float() / 255.0
            if torch.cuda.is_available():
                image_1 = image_1.cuda()

            features_1 = model(image_1.unsqueeze(0))

            for filename_2 in os.listdir(ds_2_images_dir):
                if filename_2.lower().endswith('.png'):
                    # load the image from disk
                    img_path_2 = os.path.join(ds_2_images_dir, filename_1)
                    image_2_np = cv2.imread(img_path_2, cv2.IMREAD_UNCHANGED)
                    image_2 = torch.from_numpy(image_2_np)
                    if len(image_2.shape) == 2:
                        image_2 = image_2.unsqueeze(-1).repeat((1, 1, 3))
                    image_2 = torch.permute(image_2, (2, 0, 1))  # channel dim first
                    image_2 = image_2[
                        [2, 1, 0, 3] if image_2.shape[0] == 4 else [2, 1, 0] if image_2.shape[0] == 3 else [
                            0]]  # BGR to RGB
                    image_2 = image_2[:3, ...].float() / 255.0
                    if torch.cuda.is_available():
                        image_2 = image_2.cuda()

                    features_2 = model(image_2.unsqueeze(0))

                    # use MSE as distance
                    dist = ((features_1 - features_2) ** 2.0).sum() / torch.numel(features_1)
                    if filename_1 not in dist_dict_1_2:
                        dist_dict_1_2[filename_1] = {}
                    dist_dict_1_2[filename_1][filename_2] = dist

                    if filename_2 not in dist_dict_2_1:
                        dist_dict_2_1[filename_2] = {}
                    dist_dict_2_1[filename_2][filename_1] = dist

    with open(os.path.join('dataset', dataset_1, f'sample_distances__{dataset_1}__{dataset_2}.pkl'), 'wb') as f:
        pickle.dump(dist_dict_1_2, f)

    if dataset_1 != dataset_2:
        with open(os.path.join('dataset', dataset_2, f'sample_distances__{dataset_2}__{dataset_1}.pkl'), 'wb') as f:
            pickle.dump(dist_dict_2_1, f)

    print('Finished calculating pairwise distances')


if __name__ == '__main__':
    desc_str = 'Compute and pickle the pairwise distances between samples of two datasets'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-1', '--dataset_1', required=True, type=str, help='First dataset')
    parser.add_argument('-2', '--dataset_2', required=True, type=str, help='Second dataset')
    options = parser.parse_args()
    main(options.dataset_1, options.dataset_2)
