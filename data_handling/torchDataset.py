# code taken and adapted from
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ and pytorch Data Tutorial

from torch.utils.data import Dataset
from torchvision.io import read_image


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, gt_paths=None, preprocessing=None, num_output_channels=1):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.preprocessing = preprocessing
        self.num_output_channels = num_output_channels

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.img_paths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        img_path = self.img_paths[idx]
        # load the image from disk
        image = read_image(img_path)

        # in case there is no groundTruth, only return the image
        if self.gt_paths is None:
            if self.preprocessing is not None:
                # apply the transformations to both image and its mask
                image = self.preprocessing(x=image, is_gt=False)
            return image

        # there is groundtruth
        gt = read_image(self.gt_paths[idx])
        # check to see if we are applying any transformations
        if self.preprocessing is not None:
            # apply the transformations to both image and its mask
            image = self.preprocessing(x=image, is_gt=False)
            gt = self.preprocessing(x=gt, is_gt=True, num_output_channels = self.num_output_channels)
        # return a tuple of the image and its mask
        return image, gt
