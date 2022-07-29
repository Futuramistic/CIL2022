import itertools
import tensorflow as tf
import warnings
import math
import utils
import tensorflow.keras as K
import numpy as np

from .dataloader import DataLoader


class TFDataLoader(DataLoader):
    """
    Dataloader for Tensorflow models
    """

    def __init__(self, dataset="original", pad32=False, use_geometric_augmentation=False, use_color_augmentation=False,
                 aug_contrast=[0.8, 1.2], aug_brightness=0.2, aug_saturation=[0.8, 1.2], use_adaboost=None):
        """
        Args:
            dataset (string): type of Dataset ("original" [from CIL2022 Competition], "Massachusets", ...)
                Refer to util.py for all the dataset names
            pad32 (bool): whether to pad images with zeros so that the dimensions are divisible by 32
            use_geometric_augmentation (bool): whether to augment the data on the fly with geometric augmentations
            use_color_augmentation (bool): whether to augment the data on the fly with color augmentations
            aug_contrast (list): range of values for the contrast augmentation
            aug_brightness (list): range of values for the brightness augmentation
            aug_saturation (list): range of values for the saturation augmentation
            use_adaboost (bool): whether adaboost is used
        """
        super().__init__(dataset)
        self.pad32 = pad32
        self.contrast = aug_contrast
        self.saturation = aug_saturation
        # Symmetrical double
        self.brightness = aug_brightness
        self.use_geometric_augmentation = use_geometric_augmentation
        self.use_color_augmentation = use_color_augmentation
        
        # if adaboost is used, the dataloader is used to store the current sample weights as the same dataloader is used
        # for all trainers and models
        self.use_adaboost = use_adaboost
        if self.use_adaboost:
            self.weights_set = False
            self.weights = None # create sample weights

    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split (float): training/test splitting ratio in [0,1]
        Returns: Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
                 in samples
        """
        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        unlabeled_test_size = len(self.test_img_paths)
        return train_size, test_size, unlabeled_test_size

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
        min_img = tf.zeros((1, 1, 3), dtype=tf.dtypes.uint8)
        min_img_val = tf.math.reduce_min(preprocessing(min_img, is_gt=False))
        max_img = tf.ones((1, 1, 3), dtype=tf.dtypes.uint8) * 255
        max_img_val = tf.math.reduce_max(preprocessing(max_img, is_gt=False))
        return min_img_val.numpy().item(), max_img_val.numpy().item()

    def __parse_data(self, image_path, channels=0):
        """
        Get image data
        Args:
            image_path (string): path to an image
            channels (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
        Returns:
            A Tensor of type dtype uint8
        """
        image_content = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_content, channels)

        # Pad with zeros to a size divisible by 32 - UNet-friendly; no loss of info
        if self.pad32:
            height, width = image.shape[0], image.shape[1]
            target_height, target_width = int(math.ceil(height / 32) * 32), int(math.ceil(width / 32) * 32)
            height_offset = int((target_height - height) / 2)
            width_offset = int((target_width - width) / 2)
            image = tf.image.pad_to_bounding_box(image, height_offset, width_offset, target_height, target_width)
        return image

    def __parse_data_test(self, image_path, channels=0):
        """
        Get image test data as all ones
        Args:
           image_path  (string): path to an image
           channels   (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
        Returns:
            A Tensor of type dtype uint8
        """
        return tf.ones_like(self.__parse_data(image_path, channels))

    def __get_image_data(self, img_paths, gt_paths=None, shuffle=True, preprocessing=None, offset=0, length=int(1e12),
                         return_dataloader=True):
        """
        Get images and masks
        Args:
           img_paths (list): list of image paths
           gt_paths (list): list of groundtruth paths
           shuffle (bool): whether to shuffle the images
           preprocessing: function to apply to the images
           offset (int): Offset from which we read the paths
           length (int): maximum length of the desired dataset
           return_dataloader (bool): Whether to return a dataloader or just the data
        Returns:
            Dataset or Data depending on return_dataloader
        """
        # WARNING: must use lambda captures (see https://stackoverflow.com/q/10452770)
        # WARNING: scientific notation is not recognized as an integer
        img_paths_tf = tf.convert_to_tensor(img_paths[offset:int(offset + length)])
        parse_img = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
            (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=False))
        parse_gt = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
            (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=True))

        # itertools.count() gives infinite generators

        if gt_paths is not None:
            gt_paths_tf = tf.convert_to_tensor(gt_paths[offset:offset + length])
            output_types = (parse_img(img_paths_tf[0]).dtype, parse_gt(gt_paths_tf[0]).dtype)
            if not return_dataloader:
                # don't shuffle and do return data
                return [parse_img(img_path) for _ in itertools.count() for img_path in img_paths_tf], \
                    [parse_gt(gt_path) for _ in itertools.count() for gt_path in gt_paths_tf]
            elif shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: ((parse_img(img_path), parse_gt(gt_path)) for _ in itertools.count()
                             for img_path, gt_path
                             in zip(*utils.consistent_shuffling(img_paths_tf, gt_paths_tf))),
                    output_types=output_types)
            else:
                return tf.data.Dataset.from_generator(
                    lambda: ((parse_img(img_path), parse_gt(gt_path)) for _ in itertools.count()
                             for img_path, gt_path in zip(img_paths_tf, gt_paths_tf)),
                    output_types=output_types)
        else:
            output_types = parse_img(img_paths_tf[0]).dtype
            if not return_dataloader:
                # don't shuffle and do return data
                return [(parse_img(img_path) for _ in itertools.count() for img_path in img_paths_tf)]
            elif shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: (parse_img(img_path) for _ in itertools.count() for img_path in
                             utils.consistent_shuffling(img_paths_tf)[0]),
                    output_types=output_types)
            else:
                return tf.data.Dataset.from_generator(
                    lambda: (parse_img(img_path) for _ in itertools.count() for img_path in img_paths_tf),
                    output_types=output_types)
    
    def __get_image_data_with_weighting(self, img_paths, gt_paths=None, shuffle=True, preprocessing=None, offset=0,
                                        length=int(1e12)):
        """
        Get images and masks with weights specified in self.weights (used in adaboost)
        Args:
           img_paths (list): list of image paths
           gt_paths (list): list of groundtruth paths
           shuffle (bool): whether to shuffle the images
           preprocessing: function to apply to the images
           offset (int): Offset from which we read the paths
           length (int): maximum length of the desired dataset
        Returns:
            Dataset or Data depending on return_dataloader
        """
        # WARNING: must use lambda captures (see https://stackoverflow.com/q/10452770)
        # WARNING: scientific notation is not recognized as an integer
        img_paths_tf = tf.convert_to_tensor(img_paths[offset:int(offset + length)])
        parse_img = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
            (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=False))
        parse_gt = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
            (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=True))

        # itertools.count() gives infinite generators
        
        if not self.weights_set:
            # init with equal weights
            self.weights = np.ones(length)*(1/length)
            
        # create probability disribution from weights:
        indices = list(range(length))

        if gt_paths is not None:
            gt_paths_tf = tf.convert_to_tensor(gt_paths[offset:offset + length])
            output_types = (parse_img(img_paths_tf[0]).dtype, parse_gt(gt_paths_tf[0]).dtype)
            def get_data_from_idx(sample_idx):
                img_path, gt_path = img_paths_tf[sample_idx], gt_paths_tf[sample_idx]
                return (parse_img(img_path), parse_gt(gt_path))
            
            if shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: ((get_data_from_idx(sample_idx[0])) for _ in itertools.count()
                              for sample_idx
                              in utils.consistent_shuffling(np.squeeze(np.random.choice(a=indices, size=length)))),
                    output_types=output_types)
            else:
                # note that we cannot use sample weighting without shuffling
                return tf.data.Dataset.from_generator(
                    lambda: ((get_data_from_idx(sample_idx))
                              for _ in itertools.count()
                              for sample_idx in indices),
                    output_types=output_types)
        else:
            # no weighting needed for data without groundtruth
            def get_data_from_idx(sample_idx):
                img_path = img_paths_tf[sample_idx]
                return parse_img(img_path)
            if shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: (get_data_from_idx(sample_idx[0])
                             for _ in itertools.count()
                             for sample_idx in utils.consistent_shuffling(indices)),
                    output_types=output_types)
            else:
                return tf.data.Dataset.from_generator(
                    lambda: (get_data_from_idx(sample_idx)
                              for _ in itertools.count()
                              for sample_idx in indices),
                    output_types=output_types)
                
    def get_training_data_for_one_epoch(self, split, preprocessing=None, **args):
        """
        Create training/validation dataset split, used for dynamic weighting of samples by their loss
        Args:
           split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
           preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                     constructing the native dataloader
        Returns:
            X and Y training samples
        """
        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        
        training_data_x, training_data_y = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                                 preprocessing=preprocessing, offset=0,
                                                                 length=train_size, return_dataloader=False)

        if self.use_geometric_augmentation:
            return (training_data_x,training_data_y).map(self.augmentation, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        return training_data_x, training_data_y
    
    def get_training_dataloader(self, split, batch_size, preprocessing=None, suppress_adaboost_weighting=False, **args):
        """
        Create training/validation dataset split
        Args:
           split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
           batch_size (int): training batch size
           preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                     constructing the native dataloader
            suppress_adaboost_weighting (bool): used during adaboost run: whether to ignore the given sample weights
        Returns:
            PrefetchDataset
        """

        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        # when suppress_adaboost_weighting is set to true, we are inside an adaboost run that doesn't allow shuffling
        # for this setting; suppress_adaboost_weighting is set to False for all non Adaboost runs
        shuffle = not suppress_adaboost_weighting
        if self.use_adaboost and shuffle:
            self.training_data = self.__get_image_data_with_weighting(self.training_img_paths, self.training_gt_paths,
                                                                      shuffle=shuffle, preprocessing=preprocessing, 
                                                                      offset=0, length=train_size)
            self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                                     shuffle=shuffle, preprocessing=preprocessing, 
                                                                     offset=train_size, length=test_size)
        else:
            self.training_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths, 
                                                       shuffle=shuffle, preprocessing=preprocessing, offset=0,
                                                       length=train_size)
            self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths, shuffle=False,
                                                      preprocessing=preprocessing, offset=train_size, length=test_size)
        print(f'Train data consists of ({train_size}) samples')
        print(f'Test data consists of ({test_size}) samples')

        if self.use_geometric_augmentation:
            return self.training_data.batch(batch_size).map(self.augmentation, tf.data.AUTOTUNE) \
                .prefetch(tf.data.AUTOTUNE)

        return self.training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Get labeled dataset for validation
        Args:
           batch_size (int): training batch size
           preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                     constructing the native dataloader
        Returns:
            PrefetchDataset
        """
        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. \
                Usually the test data is created by splitting the training data when calling get_training_dataloader. \
                If groundtruth test data is explicitly available in the Dataset, this will be used, otherwise the \
                complete training dataset will be used.\n \
                Call <get_unlabeled_testing_dataloader()> in order to get the test data of a dataset \
                without annotations.")

            if self.test_gt_dir is not None:
                self.testing_data = self.__get_image_data(self.test_img_paths, self.test_gt_paths, shuffle=False,
                                                          preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.test_img_paths)}) samples')
            else:
                self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                          shuffle=False, preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.training_img_paths)}) samples')

        ret = self.testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret

    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Get unlabeled dataset for test
        Args:
           batch_size (int): training batch size
           preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                     constructing the native dataloader
        Returns:
            PrefetchDataset
        """
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled test data. The test data will "
                          f"simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = self.__get_image_data(self.test_img_paths, preprocessing=preprocessing,
                                                                shuffle=False)
            print(f'Found ({len(self.test_img_paths)}) unlabeled test data')
        ret = self.unlabeled_testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        ret.img_val_min, ret.img_val_max = self.get_img_val_min_max(preprocessing)
        return ret

    def load_model(self, filepath):
        """
        Load model
        Args:
            filepath (string): Path to the saved model
        Returns:
            The loaded model
        """
        return tf.keras.models.load_model(filepath)

    def save_model(self, model, filepath):
        """
        Save model
        Args:
            model (Keras.model): The model to save
            filepath (string): Where to save the model
        """
        tf.keras.models.save_model(model, filepath)

    def augmentation(self, image, label):
        """
        Apply various data augmentations to the image
        Args:
            image: image to which we apply the augmentations
            label: the groundtruth corresponding to the image
        Returns:
            the modified image and label
        """
        seed1 = tf.random.uniform([], minval=0, maxval=pow(2, 31) - 1, dtype=tf.dtypes.int32, seed=None)
        seed2 = tf.random.uniform([], minval=0, maxval=pow(2, 31) - 1, dtype=tf.dtypes.int32, seed=None)
        seed = [seed1, seed2]
        # Flips
        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
        label = tf.image.stateless_random_flip_left_right(label, seed=seed)
        image = tf.image.stateless_random_flip_up_down(image, seed=seed)
        label = tf.image.stateless_random_flip_up_down(label, seed=seed)

        # Image colour changes
        if self.use_color_augmentation:
            img_dtype = image.dtype
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.random_brightness(image, max_delta=self.brightness, seed=None)
            image = tf.image.random_saturation(image, lower=self.saturation[0], upper=self.saturation[1], seed=None)
            image = tf.image.random_contrast(image, lower=self.contrast[0], upper=self.contrast[1], seed=None)
            image = tf.clip_by_value(image, 0.0, 0.99999)
            image = tf.image.convert_image_dtype(image, dtype=img_dtype)


        # Rotate by 90 degrees only - if we rotate by an aribitrary -> road my disappear!
        i = tf.random.uniform([], minval=0, maxval=3, dtype=tf.dtypes.int32, seed=None)
        image = tf.image.rot90(image, i)
        label = tf.image.rot90(label, i)
        return image, label
