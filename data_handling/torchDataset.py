# code taken and adapted from
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ and pytorch Data Tutorial

import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, gt_paths=None, preprocessing=None):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.preprocessing = preprocessing

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

        # check to see if we are applying any transformations
        if self.preprocessing is not None:
            # apply the transformations to both image and its mask
            image = self.preprocessing(x=image, is_gt=False)
            gt = self.preprocessing(x=gt, is_gt=True)
        # return a tuple of the image and its mask
        return image, gt
