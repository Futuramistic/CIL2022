# code taken and adapted from
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ and pytorch Data Tutorial

import cv2
import os
import random
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as FT


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, gt_paths=None, preprocessing=None, training_data_len=None,
                 use_geometric_augmentation=False, use_color_augmentation=False, contrast=[0.8,1.2], brightness=0.2,
                 saturation=[0.8,1.2], use_rl_supervision=False):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.preprocessing = preprocessing
        self.training_data_len = training_data_len
        self.use_geometric_augmentation = use_geometric_augmentation
        self.use_color_augmentation = use_color_augmentation
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.use_rl_supervision = use_rl_supervision

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.img_paths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        img_path = self.img_paths[idx]

        # load the image from disk
        image_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = torch.from_numpy(image_np)
        image = torch.permute(image, (2, 0, 1))  # channel dim first
        image = image[[2, 1, 0, 3] if image.shape[0] == 4 else [2, 1, 0] if gt.shape[0] == 3 else [0]]  # BGR to RGB

        # torchvision.io.read_image has problems with some PNG files
        # image = read_image(img_path)

        # in case there is no groundTruth, only return the image
        if self.gt_paths is None:
            if self.preprocessing is not None:
                # apply the transformations to both image and its mask
                image = self.preprocessing(x=image, is_gt=False)
            return image

        # there is groundtruth
        gt_np = cv2.imread(self.gt_paths[idx], cv2.IMREAD_UNCHANGED)
        gt = torch.from_numpy(gt_np)
        if len(gt.shape) == 3:
            gt = torch.permute(gt, (2, 0, 1))  # channel dim first
            gt = gt[[2, 1, 0, 3] if gt.shape[0] == 4 else [2, 1, 0] if gt.shape[0] == 3 else [0]]  # BGR to RGB
        else:
            gt = gt.unsqueeze(0)

        # torchvision.io.read_image has problems with some PNG files
        # gt = read_image(self.gt_paths[idx])

        if self.training_data_len not in [None, 0] and idx < self.training_data_len:
            geometric_transforms = [transforms.RandomAffine(degrees=[0, 0], translate=[0, 0], scale=[1, 1.05])]
            geometric_transforms.append(transforms.RandomVerticalFlip())
            geometric_transforms.append(transforms.RandomHorizontalFlip())
            
            lighting_transforms = [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation)]
            
            comp_geo_transform = transforms.Compose(geometric_transforms)
            comp_light_transform = transforms.Compose(lighting_transforms)

            image = image[:3]
            gt = gt[:3]

            if self.use_color_augmentation:
                image = comp_light_transform(image)
            
            if self.use_geometric_augmentation:
                # transform is random; must ensure image and GT are transformed in a consistent manner
                image, gt = torch.split(comp_geo_transform(torch.cat((image, gt), dim=0)), [image.shape[0], gt.shape[0]])
                
                degs = 90 * random.randint(-4, 4)
                image = FT.rotate(image, degs)
                gt = FT.rotate(gt, degs)

        # check to see if we are applying any transformations
        if self.preprocessing is not None:
            # apply the transformations to both image and its mask
            image = self.preprocessing(x=image, is_gt=False)
            gt = self.preprocessing(x=gt, is_gt=True)

        if not self.use_rl_supervision:
            # return a tuple of the image and its mask
            return image, gt
        else:
            # return a tuple of (the image, the optimal brush radius map, a non-maximum-suppressed version of the brush radius map) and its mask
            
            sample_filename = os.path.basename(img_path)
            sample_dir_parent_dir = os.path.dirname(os.path.dirname(img_path))
            opt_brush_radius_path = os.path.join(sample_dir_parent_dir, 'opt_brush_radius', sample_filename.replace('.png', '.pkl'))
            
            if os.path.isfile(opt_brush_radius_path):
                with open(opt_brush_radius_path, 'rb') as f:
                    opt_brush_radii = torch.from_numpy(pickle.load(f))
            else:
                opt_brush_radii = torch.zeros(image.shape[1:])
            
            non_max_supp_path = os.path.join(sample_dir_parent_dir, 'non_max_suppressed', sample_filename.replace('.png', '.pkl'))
            if os.path.isfile(non_max_supp_path):
                with open(non_max_supp_path, 'rb') as f:
                    non_max_suppressed = torch.from_numpy(pickle.load(f))
            else:
                non_max_suppressed = torch.zeros(image.shape[1:])

            return torch.cat((image, opt_brush_radii.unsqueeze(0), non_max_suppressed.unsqueeze(0)), axis=0), gt
