import warnings
import torch
from torch.utils.data import DataLoader as torchDL, Subset, WeightedRandomSampler
from .torchDataset import SegmentationDataset
from .dataloader import DataLoader
import numpy as np


class TorchDataLoader(DataLoader):
    """
    Dataloader for Torch models
    """

    def __init__(self, dataset="original", use_geometric_augmentation=False, use_color_augmentation=False,
                 aug_contrast=[0.8, 1.2], aug_brightness=[0.8, 1.2], aug_saturation=[0.8, 1.2], use_adaboost=False):
        """
        Args:
            dataset (string): type of Dataset ("original" [from CIL2022 Competition], "Massachusets", ...)
                Refer to util.py for all the dataset names
            use_geometric_augmentation (bool): whether to augment the data on the fly with geometric augmentations
            use_color_augmentation (bool): whether to augment the data on the fly with color augmentations
            aug_contrast (list): range of values for the contrast augmentation
            aug_brightness (list): range of values for the brightness augmentation
            aug_saturation (list): range of values for the saturation augmentation
            use_adaboost (bool): whether adaboost is used, therefore if data weighting is needed
        """
        super().__init__(dataset)
        self.use_geometric_augmentation = use_geometric_augmentation
        self.use_color_augmentation = use_color_augmentation
        self.contrast = aug_contrast
        self.brightness = aug_brightness
        self.saturation = aug_saturation
        
        # if adaboost is used, the dataloader is used to store the current sample weights as the same dataloader is used for all
        # trainers and models
        self.use_adaboost = use_adaboost
        if self.use_adaboost:
            self.weights = None # create sample weights
            self.weights_set = False # set to True after first adaboost run

    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/test splitting ratio in [0,1]

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
            preprocessing = lambda x, is_gt: x
        min_img = torch.zeros((3, 1, 1), dtype=torch.uint8)
        min_img_val = preprocessing(min_img, is_gt=False).min()
        max_img = torch.ones((3, 1, 1), dtype=torch.uint8) * 255
        max_img_val = preprocessing(max_img, is_gt=False).max()
        return min_img_val.detach().cpu().numpy().item(), max_img_val.detach().cpu().numpy().item()

    def get_training_dataloader(self, split, batch_size, weights=None, preprocessing=None, **kwargs):
        """
        Creates the training dataloader and also defines the testing data from the split. Therefore, call this function before
        calling "get_testing_dataloader".
        Args:
            split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
            batch_size (int): training batch size
            weights (list of floats): weights for sampling probability of each datapoint, these are not the adaboost sample weights 
                if None: don't use any weights
                if empty list: first run with equal weighting
                if not empty: list of weights for each data sample
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **kwargs: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns:
            Torch Dataloader
        """

        if 'suppress_adaboost_weighting' in kwargs:
            del kwargs['suppress_adaboost_weighting']
        
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        training_data_len = int(len(self.training_img_paths)*split)
        
        self.dataset_obj = SegmentationDataset(self.training_img_paths, self.training_gt_paths, preprocessing, training_data_len,
                                                self.use_geometric_augmentation, self.use_color_augmentation, self.contrast,
                                                self.brightness, self.saturation)
        self.training_data = Subset(self.dataset_obj, list(range(training_data_len)))
        self.testing_data = Subset(self.dataset_obj, list(range(training_data_len, len(self.dataset_obj))))

        # self.dataset is not yet set in these cases:
        if weights is None or len(weights) == 0:
            del_shuffle = 'shuffle' not in kwargs
            if del_shuffle:
                kwargs['shuffle'] = True
            ret = torchDL(self.training_data, batch_size, **kwargs)
            if del_shuffle:
                del kwargs['shuffle']
            
        elif not self.use_adaboost:
            # sample weighting (which adapts the sample weights after each epoch of the current model) is mutually exclusive with adaboost,
            # which adapts the sample weights after the model has completed the training
            sampler = WeightedRandomSampler(weights[:training_data_len], training_data_len, replacement=True)
            ret = torchDL(self.training_data, batch_size, sampler = sampler, **kwargs) # shuffling is mutually exclusive with sampler
        
        if self.use_adaboost:
            if not self.weights_set:
                # init with equal weights
                self.weights = np.ones(training_data_len, dtype=np.float16)*(1/training_data_len)
            sampler = WeightedRandomSampler(weights=self.weights[:training_data_len], num_samples=training_data_len, replacement=True)
            ret = torchDL(self.training_data, batch_size, sampler = sampler, **kwargs) # shuffling is mutually exclusive with sampler
        
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret

    def get_testing_dataloader(self, batch_size, preprocessing=None, **kwargs):
        """
        Creates the testing dataloader with groundtruth data. Call get_training_dataloader before calling this function in order to use
        the split consistently to determine the training and testing data. If no groundtruth data exists for the test dataset, the 
        training data is returned
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **kwargs: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. \
                Usually the test data is created by splitting the training data when calling get_training_dataloader. \
                If groundtruth test data is explicitely available in the Dataset, this will be used, otherwise the \
                complete training dataset will be used.\n Call <get_unlabeled_testing_dataloader()> in order \
                to get the test data of a dataset without annotations.")

            if self.test_gt_dir is not None:
                self.testing_data = SegmentationDataset(self.test_img_paths, self.test_gt_paths, preprocessing,
                                                        use_color_augmentation=False, use_geometric_augmentation=False)
            else:
                self.testing_data = SegmentationDataset(self.training_img_paths, self.training_gt_paths, preprocessing,
                                                        use_color_augmentation=False, use_geometric_augmentation=False)
        
        ret = torchDL(self.testing_data, batch_size, shuffle=False, **kwargs)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret
            
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **kwargs):
        """
        Creates the testing dataloader for testing data where either groundtruth data doesn't exist, or when groundtruth
        data is not needed.
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled test data. The test data will "
                          f"simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = SegmentationDataset(self.test_img_paths, None, preprocessing,
                                                              use_color_augmentation=False,
                                                              use_geometric_augmentation=False)
        ret = torchDL(self.unlabeled_testing_data, batch_size, shuffle=False, **kwargs)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret

    def load_model(self, path, model_class_as_string):
        """
        Load model
        Args:
            path (string): Path to the saved model
            model_class_as_string (string): Mdel class where the model parameters are loaded to
        Returns:
            The loaded model
        """
        model = eval(model_class_as_string)
        model.load_state_dict(torch.load(path))
        return model
    
    def save_model(self, model, path):
        """
        Save model
        Args:
            model (torch model): The model to save
            path (string): Where to save the model
        """
        torch.save(model.state_dict(), path)
