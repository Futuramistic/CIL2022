import os
from random import random
import warnings

import torch
from .dataloader import DataLoader
from torch.utils.data import DataLoader as torchDL
from torch.utils.data import random_split
from .torchDataset import SegmentationDataset
from models import *

class TorchDataLoader(DataLoader):
    def __init__(self, dataset="original"):
        super().__init__(dataset)
    
    def get_training_dataloader(self, split, batch_size, **args):
        """
        Args:
            split (float): training/testing splitting ratio, e.g. 0.8 for 80"%" training and 20"%" testing data
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns:
            Torch Dataloader
        """
        #load training data and possibly split
        dataset = SegmentationDataset(self.img_dir, self.gt_dir)
        training_data_len = (int) (len(dataset)*split)
        testing_data_len = len(dataset)-training_data_len
        
        self.training_data, self.testing_data = random_split(dataset, [training_data_len, testing_data_len])
        return torchDL(self.training_data, batch_size, **args)
    
    def get_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        if self.testing_data is None:
            warnings.warn("You called testing dataloader before training dataloader. \
                Usually the testing data is created by splitting the training data when calling get_training_dataloader. \
                    If groundtruth testing data is explicitely available in the Dataset, this will be used, otherwise the complete training dataset will be used.\n \
                        Call <get_unlabeled_testing_dataloader()> in order to get the testing data of a dataset without annotations.")
            if self.test_gt_dir is not None:
                self.testing_data = SegmentationDataset(self.test_img_dir,self.test_gt_dir)
            else:
                self.testing_data = SegmentationDataset(self.img_dir, self.gt_dir)
        else:
            return torchDL(self.testing_data, batch_size, **args)
            
    def get_unlabeled_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled testing data. The testing data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = SegmentationDataset(self.test_img_dir)
        return torchDL(self.unlabeled_testing_data, batch_size, **args)
        
    
    def load_model(self, path, model_class_as_string):
        model = eval(model_class_as_string)
        model.load_state_dict(torch.load(path))
        return model
        
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)