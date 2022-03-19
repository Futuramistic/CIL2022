import itertools
import tensorflow as tf
import warnings

from .dataloader import DataLoader
import utils


class TFDataLoader(DataLoader):
    def __init__(self, dataset="original"):
        super().__init__(dataset)

    # Get image data
    # Args:
    #    image_path  (string): path to an image
    #    channels   (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
    # Returns: A Tensor of type dtype uint8
    ###  
    def __parse_data(self, image_path, channels=0):
        image_content = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_content, channels)
        return image

    # Get images and masks
    # Args:
    #    image_dir  (string): the directory of images
    #    mask_dir   (string): the directory of corresponding masks
    # Returns: Dataset
    ###  
    def __get_image_data(self, img_paths, gt_paths=None, preprocessing=None, offset=0, length=10**12):
        # WARNING: must use lambda captures (see https://stackoverflow.com/q/10452770)
        img_paths_tf = tf.convert_to_tensor(img_paths[offset:offset+length])
        parse = (lambda x, preprocessing=preprocessing: self.__parse_data(x)) if preprocessing is None else \
                (lambda x, preprocessing=preprocessing: preprocessing(self.__parse_data(x)))

        # itertools.count() gives infinite generators

        if gt_paths is not None:
            gt_paths_tf = tf.convert_to_tensor(gt_paths[offset:])
            common_gen = ((parse(img_path), parse(gt_path)) for _ in itertools.count() for img_path, gt_path in
                          zip(*utils.consistent_shuffling(img_paths_tf, gt_paths_tf)))
            return tf.data.Dataset.from_generator(lambda common_gen=common_gen: common_gen,
                                                  output_types=(parse(img_paths_tf[0]).dtype,
                                                                parse(gt_paths_tf[0]).dtype))
        else:
            image_gen = (parse(img_path) for _ in itertools.count() for img_path in
                         utils.consistent_shuffling(img_paths_tf)[0])
            return tf.data.Dataset.from_generator(lambda image_gen=image_gen: image_gen,
                                                  output_types=parse(img_paths_tf[0]).dtype)

    ###
    # Create training/validaiton dataset split
    # Args:
    #    split (float): training/testing splitting ratio, e.g. 0.8 for 80"%" training and 20"%" testing data
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: PrefetchDataset
    ###  
    def get_training_dataloader(self, split, batch_size, preprocessing=None, **args):
        # Get images' names and data
        dataset_size = len(self.training_img_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        self.training_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                   preprocessing=preprocessing, offset=0, length=train_size)
        self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                  preprocessing=preprocessing, offset=train_size, length=test_size)
        print(f'Train data consists of ({train_size}) samples')
        print(f'Test data consists of ({test_size}) samples')

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
            warnings.warn("You called testing dataloader before training dataloader. \
                Usually the testing data is created by splitting the training data when calling get_training_dataloader. \
                    If groundtruth testing data is explicitely available in the Dataset, this will be used, otherwise the complete training dataset will be used.\n \
                        Call <get_unlabeled_testing_dataloader()> in order to get the testing data of a dataset without annotations.")
            if self.test_gt_dir is not None:
                self.testing_data = self.__get_image_data(self.test_img_paths, self.test_gt_paths,
                                                          preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.test_img_paths)}) samples')
            else:
                self.testing_data = self.__get_image_data(self.training_img_paths, self.training_gt_paths,
                                                          preprocessing=preprocessing)
                print(f'Test data consists of ({len(self.training_img_paths)}) samples')
        return self.testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # return self.testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    ###
    # Get unlabeled dataset for testing
    # Args:
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: PrefetchDataset
    ###       
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled testing data. The testing data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = self.__get_image_data(self.test_img_paths, preprocessing=preprocessing)
            print(f'Found ({len(self.test_img_paths)}) unlabeled testing data')
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
