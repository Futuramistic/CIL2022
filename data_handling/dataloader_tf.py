import itertools
from random import randint
import tensorflow as tf
import warnings
import math
from .dataloader import DataLoader
import utils
import tensorflow_addons as tfa

class TFDataLoader(DataLoader):

    def __init__(self, dataset="original", pad32 = False, use_augemntation = False):
        super().__init__(dataset)
        self.pad32 = pad32
        self.use_augmentation = use_augemntation

    # Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
    # Args:
    #   split   (float): training/test splitting ratio \in [0,1]
    # Returns: Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
    #          in samples
    def get_dataset_sizes(self, split):
        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        unlabeled_test_size = len(self.test_img_paths)
        return train_size, test_size, unlabeled_test_size


    # Get image data
    # Args:
    #    image_path  (string): path to an image
    #    channels   (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
    # Returns: A Tensor of type dtype uint8
    ###  
    def __parse_data(self, image_path, channels=0):
        image_content = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_content, channels)

        # Pad with zeros to a size divisible by 32 - UNet-friendly; no loss of info
        if self.pad32:
            height, width = image.shape[0],image.shape[1]
            target_height, target_width = int(math.ceil(height/32)*32), int(math.ceil(width/32)*32)
            height_offset = int((target_height-height)/2)
            width_offset = int((target_width-width)/2)
            image = tf.image.pad_to_bounding_box(image,height_offset,width_offset,target_height,target_width)
        return image

    # Get image test data as all ones
    # Args:
    #    image_path  (string): path to an image
    #    channels   (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
    # Returns: A Tensor of type dtype uint8
    ###  
    def __parse_data_test(self, image_path, channels=0):
        return tf.ones_like(self.__parse_data(image_path,channels))

    # Get images and masks
    # Args:
    #    image_dir  (string): the directory of images
    #    mask_dir   (string): the directory of corresponding masks
    # Returns: Dataset
    ###  
    def __get_image_data(self, img_paths, gt_paths=None, shuffle=True, preprocessing=None, offset=0, length=1e12):
        # WARNING: must use lambda captures (see https://stackoverflow.com/q/10452770)
        img_paths_tf = tf.convert_to_tensor(img_paths[offset:int(offset+length)])  # scientific notation is not recognized as an integer
        parse_img = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
                    (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=False))
        parse_gt = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
                   (lambda x, preprocessing=preprocessing: preprocessing(x=self.__parse_data(x), is_gt=True))

        # itertools.count() gives infinite generators

        if gt_paths is not None:
            gt_paths_tf = tf.convert_to_tensor(gt_paths[offset:offset+length])
            output_types = (parse_img(img_paths_tf[0]).dtype, parse_gt(gt_paths_tf[0]).dtype)
            if shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: ((parse_img(img_path), parse_gt(gt_path)) for _ in itertools.count()
                             for img_path, gt_path in zip(*utils.consistent_shuffling(img_paths_tf, gt_paths_tf))),
                    output_types=output_types)
            else:
                return tf.data.Dataset.from_generator(
                    lambda: ((parse_img(img_path), parse_gt(gt_path)) for _ in itertools.count()
                             for img_path, gt_path in zip(img_paths_tf, gt_paths_tf)),
                    output_types=output_types)
        else:
            output_types = parse_img(img_paths_tf[0]).dtype
            if shuffle:
                return tf.data.Dataset.from_generator(
                    lambda: (parse_img(img_path) for _ in itertools.count() for img_path in
                             utils.consistent_shuffling(img_paths_tf)[0]),
                    output_types=output_types)
            else:
                return tf.data.Dataset.from_generator(
                    lambda: (parse_img(img_path) for _ in itertools.count() for img_path in img_paths_tf),
                    output_types=output_types)

    ###
    # Create training/validaiton dataset split
    # Args:
    #    split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: PrefetchDataset
    ###  
    def get_training_dataloader(self, split, batch_size, preprocessing=None, **args):
        # Get images' names and data

        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size

        self.training_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths, shuffle=True,
                                                   preprocessing=preprocessing, offset=0, length=train_size)
        self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths, shuffle=False,
                                                  preprocessing=preprocessing, offset=train_size, length=test_size)
        print(f'Train data consists of ({train_size}) samples')
        print(f'Test data consists of ({test_size}) samples')

        if self.use_augmentation:
            return self.training_data.shuffle(50, reshuffle_each_iteration=True).map(self.augmentation,tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return self.training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # # Shuffle and split
        # data = data.shuffle(100)
        # self.training_data = data.take(train_size)
        # self.testing_data = data.skip(train_size)
        # # Cache training -> shuffle -> batch -> repeat for all batches -> prefetch for speed
        # return self.training_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    ###
    # Get labeled dataset for validation
    # Args:
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: PrefetchDataset
    ###       
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. \
                Usually the test data is created by splitting the training data when calling get_training_dataloader. \
                    If groundtruth test data is explicitely available in the Dataset, this will be used, otherwise the complete training dataset will be used.\n \
                        Call <get_unlabeled_testing_dataloader()> in order to get the test data of a dataset without annotations.")
            if self.test_gt_dir is not None:
                self.testing_data = self.__get_image_data(self.test_img_paths, self.test_gt_paths, shuffle=False,
                                                          preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.test_img_paths)}) samples')
            else:
                self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                          shuffle=False, preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.training_img_paths)}) samples')
        return self.testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # return self.testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    ###
    # Get unlabeled dataset for test
    # Args:
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: PrefetchDataset
    ###       
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled test data. The test data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = self.__get_image_data(self.test_img_paths, preprocessing=preprocessing,
                                                                shuffle=False)
            print(f'Found ({len(self.test_img_paths)}) unlabeled test data')
        return self.unlabeled_testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # return self.unlabeled_testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        
    # Load model
    # Args:
    #   filepath (string)
    def load_model(self, filepath):
        return tf.keras.models.load_model(filepath)
        
    # Save model
    # Args:
    #   model (Keras.model)
    #   filepath (string)
    def save_model(self, model, filepath):
        tf.keras.models.save_model(model,filepath)

    # Flip the image randomly (add possible augmentations later)
    def augmentation(self,image,label):
        seed1 = randint(0,pow(2,31)-1)
        seed2 = randint(0,pow(2,31)-1)
        seed = [seed1,seed2]

        # Flips
        image = tf.image.stateless_random_flip_left_right(image,seed=seed)
        label = tf.image.stateless_random_flip_left_right(label,seed=seed)
        image = tf.image.stateless_random_flip_up_down(image,seed=seed)
        label = tf.image.stateless_random_flip_up_down(label,seed=seed)

        # Image colour changes
        image = tf.image.stateless_random_brightness(image,max_delta=0.2,seed=seed)
        image = tf.image.stateless_random_saturation(image,lower=0.8,upper=1.2,seed=seed)
        image = tf.image.stateless_random_contrast(image,lower=0.8,upper=1.2,seed=seed)
        image = tf.clip_by_value(image,0,1)

        # Rotate by 90 degrees only - if we rotate by an aribitrary -> road my disappear!
        angles = [0.0,90.0,180.0,270.0]
        i = randint(0,4)
        if(i!=0):
            image = tfa.image.rotate(image,angles[i]*math.pi/180.0,interpolation='bilinear')
            label = tfa.image.rotate(label,angles[i]*math.pi/180.0,interpolation='bilinear')
        return image, label