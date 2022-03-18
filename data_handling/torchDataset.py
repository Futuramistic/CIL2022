# code taken and adapted from
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ and pytorch Data Tutorial

from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from utils import ACCEPTED_IMAGE_EXTENSIONS


class SegmentationDataset(Dataset):
    def __init__(self, imagePath, groundTruthPath=None, preprocessing=None):
        self.imagePaths = []
        for imgName in os.listdir(imagePath):
            _, ext = os.path.splitext(imgName)
            if ext.lower() in ACCEPTED_IMAGE_EXTENSIONS:
                pth = f"{imagePath}/{imgName}"
                self.imagePaths.append(pth)
        if groundTruthPath is not None:
            self.groundTruthPaths = [];
            for imgName in os.listdir(groundTruthPath):
                _, ext = os.path.splitext(imgName)
                if ext.lower() in ACCEPTED_IMAGE_EXTENSIONS:
                    pth = f"{groundTruthPath}/{imgName}"
                    self.groundTruthPaths.append(pth)
        else: self.groundTruthPaths = None
        self.preprocessing = preprocessing
  
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePaths = self.imagePaths[idx]
        # load the image from disk
        image = read_image(imagePaths)
        
        # in case there is no groundTruth, only return the image
        if self.groundTruthPaths is None:
            if self.preprocessing is not None:
                # apply the transformations to both image and its mask
                image = self.preprocessing(image)
            return image
        
        # there is groundtruth
        groundTruth = read_image(self.groundTruthPaths[idx])
        # check to see if we are applying any transformations
        if self.preprocessing is not None:
            # apply the transformations to both image and its mask
            image = self.preprocessing(image)
            groundTruth = self.preprocessing(groundTruth)
        # return a tuple of the image and its mask
        return (image, groundTruth)
