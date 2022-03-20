import warnings
import torch
from torch.utils.data import DataLoader as torchDL
from torch.utils.data import random_split
from .torchDataset import SegmentationDataset
from .dataloader import DataLoader
import utils
from models import *


class TorchDataLoader(DataLoader):
    def __init__(self, dataset="original"):
        super().__init__(dataset)

    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, testing and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/testing splitting ratio \in [0,1]

        Returns:
            Tuple of (int, int, int): sizes of training, testing and unlabeled testing datasets, respectively,
            in samples
        """
        full_training_data_len = len(self.training_img_paths)
        training_data_len = int(full_training_data_len * split)
        testing_data_len = full_training_data_len - training_data_len
        unlabeled_testing_data_len = len(self.test_img_paths)
        return training_data_len, testing_data_len, unlabeled_testing_data_len

    def get_training_dataloader(self, split, batch_size, preprocessing=None, **args):
        """
        Args:
            split (float): training/testing splitting ratio, e.g. 0.8 for 80"%" training and 20"%" testing data
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns:
            Torch Dataloader
        """
        #load training data and possibly split
        shuffled_training_img_paths, shuffled_training_gt_paths = utils.consistent_shuffling(self.training_img_paths,
                                                                                             self.training_gt_paths)
        dataset = SegmentationDataset(shuffled_training_img_paths, shuffled_training_gt_paths, preprocessing)
        training_data_len = int(len(dataset)*split)
        testing_data_len = len(dataset)-training_data_len
        
        self.training_data, self.testing_data = random_split(dataset, [training_data_len, testing_data_len])
        return torchDL(self.training_data, batch_size, shuffle=True, **args)
    
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
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
                shuffled_test_img_paths, shuffled_test_gt_paths = utils.consistent_shuffling(self.test_img_paths,
                                                                                             self.test_gt_paths)
                self.testing_data = SegmentationDataset(shuffled_test_img_paths, shuffled_test_gt_paths, preprocessing)
            else:
                shuffled_training_img_paths, shuffled_training_gt_paths =\
                    utils.consistent_shuffling(self.training_img_paths, self.training_gt_paths)
                self.testing_data = SegmentationDataset(shuffled_training_img_paths, shuffled_training_gt_paths,
                                                        preprocessing)

        return torchDL(self.testing_data, batch_size, shuffle=False, **args)
            
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled testing data. The testing data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = SegmentationDataset(*utils.consistent_shuffling(self.test_img_paths), None,
                                                              preprocessing)
        return torchDL(self.unlabeled_testing_data, batch_size, shuffle=False, **args)

    def load_model(self, path, model_class_as_string):
        model = eval(model_class_as_string)
        model.load_state_dict(torch.load(path))
        return model
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
