import os
from pkgutil import get_data
from random import random
import warnings

import tensorflow as tf


from .dataloader import DataLoader
from models import *

AUTOTUNE = tf.data.AUTOTUNE

class TFDataLoader(DataLoader):
    def __init__(self, dataset="original"):
        super().__init__(dataset)

    def parse_data(self,image_path, channels):
        image_content = tf.io.read_file(image_path)
        image  = tf.image.decode_png(image_content, channels=channels)
        return image

    def get_image_data(self,image_dir,mask_dir):
        image_paths = tf.convert_to_tensor([image_dir+"/"+x for x in tf.io.gfile.listdir(image_dir)])
        images  = [self.parse_data(x,3) for x in image_paths]
        mask_paths  = tf.convert_to_tensor([mask_dir+"/"+x for x in tf.io.gfile.listdir(mask_dir)])
        masks   = [self.parse_data(x,1) for x in mask_paths]
        return tf.data.Dataset.from_tensor_slices((images,masks))

    def get_training_dataloader(self, split, batch_size, **args):
        """
        Args:
            split (float): training/testing splitting ratio, e.g. 0.8 for 80"%" training and 20"%" testing data
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns:
            PrefetchDataset
        """
        # Get image names and recover the image data

        # Split
        data = self.get_image_data(self.img_dir,self.gt_dir)
        train_size = int(len(data) * split)
        # Shuffle
        data = data.shuffle(100)
        # Split
        self.training_data = data.take(train_size)
        self.testing_data = data.skip(train_size)
        print(f'Train data consists of ({len(self.training_data)}) samples')
        print(f'Test data consists of ({len(self.testing_data)}) samples')
        # Cache training -> shuffle -> batch -> repeat for all batches -> prefetch for speed
        return self.training_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    def get_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns: PrefetchDataset
        """
        if self.testing_data is None:
            warnings.warn("You called testing dataloader before training dataloader. \
                Usually the testing data is created by splitting the training data when calling get_training_dataloader. \
                    If groundtruth testing data is explicitely available in the Dataset, this will be used, otherwise the complete training dataset will be used.\n \
                        Call <get_unlabeled_testing_dataloader()> in order to get the testing data of a dataset without annotations.")
            if self.test_gt_dir is not None:
                self.testing_data = self.get_image_data(self.test_img_dir,self.test_gt_dir)
            else:
                self.testing_data = self.get_image_data(self.img_dir, self.gt_dir)
            print(f'Test data consists of ({len(self.testing_data)}) samples')
            return self.testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        else:
            print(f'Test data consists of ({len(self.testing_data)}) samples')
            return self.testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
            
    def get_unlabeled_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
        Returns: PrefetchDataset
        """
        if self.test_gt_dir is not None:
            warnings.warn(f"The dataset {self.dataset} doesn't contain unlabeled testing data. The testing data will simply be used without loading the groundtruth")
        if self.unlabeled_testing_data is None:
            image_paths = tf.convert_to_tensor([self.test_img_dir+"/"+x for x in tf.io.gfile.listdir(self.test_img_dir)])
            images  = [self.parse_data(x,3) for x in image_paths]
            self.unlabeled_testing_data = tf.data.Dataset.from_tensor_slices(images)
        print(f'Found ({len(self.unlabeled_testing_data)}) unlabeled testing data')
        return self.unlabeled_testing_data.cache().shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        
    
    def load_model(self, filepath):
        return tf.keras.models.load_model(filepath)
        
    
    def save_model(self, model, filepath):
        tf.keras.models.save_model(model,filepath)
