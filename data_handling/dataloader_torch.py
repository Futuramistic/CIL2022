import warnings
import torch
from torch.utils.data import DataLoader as torchDL, Dataset, Subset
from .torchDataset import SegmentationDataset
from .dataloader import DataLoader
from torchvision import transforms
import utils
from models import *


class TorchDataLoader(DataLoader):
    def __init__(self, dataset="original", use_geometric_augmentation=False, use_color_augmentation=False,
                 aug_contrast=[0.8,1.2], aug_brightness=[0.8, 1.2], aug_saturation=[0.8,1.2]):
        super().__init__(dataset)
        self.use_geometric_augmentation = use_geometric_augmentation
        self.use_color_augmentation = use_color_augmentation
        self.contrast = aug_contrast
        self.brightness = aug_brightness
        self.saturation = aug_saturation

    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/test splitting ratio \in [0,1]

        Returns:
            Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
            in samples
        """
        full_training_data_len = len(self.training_img_paths)
        training_data_len = int(full_training_data_len * split)
        testing_data_len = full_training_data_len - training_data_len
        unlabeled_testing_data_len = len(self.test_img_paths)
        return training_data_len, testing_data_len, unlabeled_testing_data_len

    def get_img_val_min_max(self, preprocessing):
        """
        Get the minimum and maximum possible values of an image pixel, when preprocessing the image using the given
        preprocessing function.
        Args:
            preprocessing: function taking a raw sample and returning a preprocessed sample to be used when
                           constructing the native dataloader
        Returns:
            Tuple of (int, int)
        """
        if preprocessing is None:
            preprocessing = lambda x: x
        min_img = torch.zeros((3, 1, 1), dtype=torch.uint8)
        min_img_val = preprocessing(min_img, is_gt=False).min()
        max_img = torch.ones((3, 1, 1), dtype=torch.uint8) * 255
        max_img_val = preprocessing(max_img, is_gt=False).max()
        return min_img_val.detach().cpu().numpy().item(), max_img_val.detach().cpu().numpy().item()

    def get_training_dataloader(self, split, batch_size, preprocessing=None, **args):
        """
        Args:
            split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns:
            Torch Dataloader
        """
        #load training data and possibly split
        
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        training_data_len = int(len(self.training_img_paths)*split)
        testing_data_len = len(self.training_img_paths)-training_data_len

        dataset = SegmentationDataset(self.training_img_paths, self.training_gt_paths, preprocessing, training_data_len,
                                      self.use_geometric_augmentation, self.use_color_augmentation, self.contrast,
                                      self.brightness, self.saturation)

        self.training_data = Subset(dataset, list(range(training_data_len)))
        self.testing_data = Subset(dataset, list(range(training_data_len, len(dataset))))
        
        ret = torchDL(self.training_data, batch_size, shuffle=True, **args)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret
        
    
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
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. \
                Usually the test data is created by splitting the training data when calling get_training_dataloader. \
                    If groundtruth test data is explicitely available in the Dataset, this will be used, otherwise the complete training dataset will be used.\n \
                        Call <get_unlabeled_testing_dataloader()> in order to get the test data of a dataset without annotations.")
            if self.test_gt_dir is not None:
                self.testing_data = SegmentationDataset(self.test_img_paths, self.test_gt_paths, preprocessing,
                                                        use_color_augmentation=False, use_geometric_augmentation=False)
            else:
                self.testing_data = SegmentationDataset(self.training_img_paths, self.training_gt_paths, preprocessing,
                                                        use_color_augmentation=False, use_geometric_augmentation=False)
        
        ret = torchDL(self.testing_data, batch_size, shuffle=False, **args)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret
            
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
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled test data. The test data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            # self.unlabeled_testing_data = SegmentationDataset(*utils.consistent_shuffling(self.test_img_paths), None,
            #                                                   preprocessing)
            self.unlabeled_testing_data = SegmentationDataset(self.test_img_paths, None, preprocessing,
                                                              use_color_augmentation=False,
                                                              use_geometric_augmentation=False)
        ret = torchDL(self.unlabeled_testing_data, batch_size, shuffle=False, **args)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret

    def load_model(self, path, model_class_as_string):
        model = eval(model_class_as_string)
        model.load_state_dict(torch.load(path))
        return model
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)