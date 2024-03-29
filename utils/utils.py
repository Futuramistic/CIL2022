"""
! DO NOT MOVE ! (because of the ROOT_DIR value)

This file contains global variables, Information on the Datasets and MlFlow, as well as usefull helper functions used
throughout the framework.
"""

import math
import os
import pathlib
import random
import sys
import time
import torch

from stat import S_ISDIR, S_ISREG

###########################################################################################
##################################    global constants    #################################
###########################################################################################

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go one dir up from the dir this file is in
SESSION_ID = int(time.time() * 1000)  # import time of utils.py in milliseconds will be the session ID
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace', lambda: None)() is not None
ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif"]
DEFAULT_SEGMENTATION_THRESHOLD = 0.5
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_NUM_SAMPLES_TO_VISUALIZE = 36
DEFAULT_BLOBS_REMOVAL_THRESHOLD = 100
DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT = 0.88  # refers to weighted F1 score
DEFAULT_TF_INPUT_SHAPE = (400, 400, 3)
OFFSET_FOR_SAVING_IMAGES = 144

# Contains the URLs where our datasets are stored, as well as a short description for each dataset
DATASET_ZIP_URLS = {
    # "original": dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
    "original": "https://polybox.ethz.ch/index.php/s/x2RcSv4MOG3rtPB/download",

    # validation split 1 of original dataset (samples "satimage_0.png" to "satimage_24.png" from "original" dataset
    # used as validation set)
    "original_split_1": "https://polybox.ethz.ch/index.php/s/EhNndrS2fIWfWZF/download",
    
    # validation split 2 of original dataset (samples "satimage_25.png" to "satimage_49.png" from "original" dataset
    # used as validation set)
    "original_split_2": "https://polybox.ethz.ch/index.php/s/TKzv8THbJPEdH9i/download",
    
    # validation split 3 of original dataset (samples "satimage_50.png" to "satimage_74.png" from "original" dataset
    # used as validation set)
    "original_split_3": "https://polybox.ethz.ch/index.php/s/eQdRlIKlIGJ7EWg/download",

    # "ext_original": "original" dataset, extended with 80 images scraped from Google Maps
    "ext_original": "https://polybox.ethz.ch/index.php/s/mj4aokQ7ZMouMyh/download",

    # "new_original": "original" dataset, with first 25 samples moved to end to form the validation split
    # same 25 samples as in "new_ext_original", "new_original_aug_6" and "ext_original_aug_6" datasets
    # use split of 0.827 to use exactly these 25 samples as the validation set
    "new_original": "https://polybox.ethz.ch/index.php/s/1l67z55lmemASnb/download",

    # "new_ext_original": "ext_original" dataset, with first 25 samples moved to end to form the validation split
    # same 25 samples as in "new_original", "new_original_aug_6" and "ext_original_aug_6" datasets
    # use split of 0.89 to use exactly these 25 samples as the validation set
    "new_ext_original": "https://polybox.ethz.ch/index.php/s/GAv6JhORUjZOq5U/download",

    # "new_ext_original_oversampled": "ext_original" dataset, with second city class oversampled, and first 25 samples
    #  moved to end to form the validation split same 25 samples as in "new_original", "new_original_aug_6"
    # and "ext_original_aug_6" datasets
    # use split of 0.917 to use exactly these 25 samples as the validation set
    "new_ext_original_oversampled": "https://polybox.ethz.ch/index.php/s/hC4bkjF7PcPGtpL/download",

    # "original_gt": dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition, but with
    # images replaced by ground truth
    "original_gt": "https://polybox.ethz.ch/index.php/s/kORjGAbqFvjG4My/download",

    # "original_128": "original" dataset, patchified into 128x128 patches and augmented using Preprocessor
    # WARNING: take into account that this dataset has 2160/720 train/"unlabeled test" images
    # (original has only 144/144, resp.)
    # e.g. use fewer epochs and a high train fraction (high "split" value), so the evaluation doesn't take too long!
    "original_128": "https://polybox.ethz.ch/index.php/s/c68pfFLBXCCjzDT/download",

    # "original_256": "original" dataset, patchified into 256x256 patches and augmented using Preprocessor
    # WARNING: take into account that this dataset has 2160/720 train/"unlabeled test" images (original
    # has only 144/144, resp.)
    # e.g. use fewer epochs and a high train fraction (high "split" value), so the evaluation doesn't take too long!
    "original_256": "https://polybox.ethz.ch/index.php/s/ncSp9vsJ1HAIHcR/download",

    # "additional_maps_1": dataset retrieved from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/ -> maps.tar.gz
    # and processed to convert the RGB masks to B/W. The resulting masks are not perfect and could definitely
    # use some better processing, but they seem to be good enough visually
    "new_maps": "https://polybox.ethz.ch/index.php/s/QTxb24YMpL1Rs66/download",

    # Massachusetts Road dataset (256x256)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 25,328 Images
    # Testing: 1,293 Images
    # WARNING: some images have white or black outer values due to processing (mostly bottom or right side)
    "massachusetts_256": "https://polybox.ethz.ch/index.php/s/WnctKQV89H6W7KT/download",

    # Massachusetts Road dataset (128x128)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 81,669 Images
    # Testing: 4,176 Images
    # WARNING: some images have white or black outer values due to processing (mostly bottom or right side)
    "massachusetts_128": "https://polybox.ethz.ch/index.php/s/XjSto2pXCeZydiH/download",

    # Massachusetts Road dataset (400x400) + the original (400x400)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 12,982 Images - Massachusetts (testing+training) + original trainig
    # Testing: 144 Images - only original testing images
    # WARNING: Some images have white or black outer values due to processing (mostly bottom or right side)!
    #          However, the number of "partial" images is limited
    "large": "https://polybox.ethz.ch/index.php/s/uXJgQrQazhrn5gA/download",

    # "original_aug_6": "original" dataset, 400x400 but with augmented training set using Preprocessor (x6)
    # I usually use a 0.975 split for this dataset
    "original_aug_6": "https://polybox.ethz.ch/index.php/s/ICjaUr4ayCNwySJ/download",

    # Recreation of "original_aug_6" dataset, but with 25 samples from original dataset excluded from augmentation
    # procedure to avoid data leakage; same 25 samples as in "new_original", "new_ext_original" and
    # "ext_original_aug_6" datasets
    # use split of 0.971 to use exactly these 25 samples as the validation set
    "new_original_aug_6": "https://polybox.ethz.ch/index.php/s/LJZ0InoG6GwyGsC/download",

    # Recreation of "original_aug_6" dataset, but with 80 additional samples scraped from Google Maps added before
    # augmentation procedure, and with 25 samples from original dataset excluded from augmentation procedure
    # to avoid data leakage; same 25 samples as in "new_original", "new_ext_original" and "new_original_aug_6" datasets
    # use split of 0.9825 to use exactly these 25 samples as the validation set
    "ext_original_aug_6": "https://polybox.ethz.ch/index.php/s/9hDXLlX7mB5Xljq/download",

    # validation split 1 of original dataset (samples "satimage_0.png" to "satimage_24.png" from "original" dataset
    # used as validation set)
    "original_split_1": "https://polybox.ethz.ch/index.php/s/EhNndrS2fIWfWZF/download",
    
    # validation split 2 of original dataset (samples "satimage_25.png" to "satimage_49.png" from "original" dataset
    # used as validation set)
    "original_split_2": "https://polybox.ethz.ch/index.php/s/TKzv8THbJPEdH9i/download",
    
    # validation split 3 of original dataset (samples "satimage_50.png" to "satimage_74.png" from "original" dataset
    # used as validation set)
    "original_split_3": "https://polybox.ethz.ch/index.php/s/eQdRlIKlIGJ7EWg/download",
    
    # validation split 2 of original dataset (samples "satimage_25.png" to "satimage_49.png" from "original" dataset used as validation set), with augmented training set using Preprocessor (x6)
    "original_split_2_aug_6": "https://polybox.ethz.ch/index.php/s/aMHd9GlkUcxOpjS/download",
    
    # validation split 3 of original dataset (samples "satimage_50.png" to "satimage_74.png" from "original" dataset used as validation set), with augmented training set using Preprocessor (x6)
    "original_split_3_aug_6": "https://polybox.ethz.ch/index.php/s/B4kB2bNmkkp4Fl2/download",

    # hand-filtered dataset of 1597 satellite images screenshotted from Google Maps
    # same 25 validation samples as in "new_original", "new_ext_original" and
    # "new_original_aug_6" datasets; use split of 0.9846 to use exactly these 25 samples as the validation set
    "maps_filtered": "https://polybox.ethz.ch/index.php/s/MfCcVyZRJ6TRDWb/download",
    
    # maps_filtered, without any original samples from the training set,
    # 400x400 but with augmented training set using Preprocessor (x6)
    # same 25 validation samples as in "new_original", "maps_filtered", etc.
    # use split of 0.9977 to use exactly these 25 samples as the validation set
    "maps_filtered_no_original_aug_6": "https:,//polybox.ethz.ch/index.php/s/o0sJxeuujwPWbxH/download",
    
    # maps_filtered, with 119 samples from the training set,
    # 400x400 but with augmented training set using Preprocessor (x6)
    # same 25 validation samples as in "new_original", "maps_filtered", etc.
    # use split of 0.9978 to use exactly these 25 samples as the validation set
    "maps_filtered_aug_6": "https://polybox.ethz.ch/index.php/s/UgTKjrRImltvLQl/download"
}
DATASET_STATS = {
'original': {
    'pixel_mean_0': 161.983, 'pixel_mean_1': 162.134, 'pixel_mean_2': 162.231,
    'pixel_std_0': 72.398, 'pixel_std_1': 72.468, 'pixel_std_2': 72.195},
'new_original': {
    'pixel_mean_0': 161.983, 'pixel_mean_1': 162.134, 'pixel_mean_2': 162.231,
    'pixel_std_0': 72.398, 'pixel_std_1': 72.468, 'pixel_std_2': 72.195},
'new_ext_original': {
    'pixel_mean_0': 156.617, 'pixel_mean_1': 156.717, 'pixel_mean_2': 156.736,
    'pixel_std_0': 73.563, 'pixel_std_1': 73.595, 'pixel_std_2': 73.470},
'new_original_aug_6': {
    'pixel_mean_0': 97.204, 'pixel_mean_1': 103.475, 'pixel_mean_2': 108.853,
    'pixel_std_0': 95.648, 'pixel_std_1': 95.778, 'pixel_std_2': 95.376},
'original_aug_6': {
    'pixel_mean_0': 151.766, 'pixel_mean_1': 153.037, 'pixel_mean_2': 154.239,
    'pixel_std_0': 83.284, 'pixel_std_1': 82.505, 'pixel_std_2': 81.720},
'ext_original_aug_6': {
    'pixel_mean_0': 94.396, 'pixel_mean_1': 100.863, 'pixel_mean_2': 106.722,
    'pixel_std_0': 93.997, 'pixel_std_1': 94.120, 'pixel_std_2': 93.762},
'new_ext_original_oversampled': {
    'pixel_mean_0': 161.241, 'pixel_mean_1': 161.431, 'pixel_mean_2': 161.521,
    'pixel_std_0': 72.192, 'pixel_std_1': 72.179, 'pixel_std_2': 71.944},
'original_gt': {
    'pixel_mean_0': 0.120, 'pixel_mean_1': 0.120, 'pixel_mean_2': 0.120,
    'pixel_std_0': 5.551, 'pixel_std_1': 5.545, 'pixel_std_2': 5.536},
'original_split_1': {
    'pixel_mean_0': 161.983, 'pixel_mean_1': 162.134, 'pixel_mean_2': 162.231,
    'pixel_std_0': 72.398, 'pixel_std_1': 72.468, 'pixel_std_2': 72.195},
'original_split_2': {
    'pixel_mean_0': 161.983, 'pixel_mean_1': 162.134, 'pixel_mean_2': 162.231,
    'pixel_std_0': 72.398, 'pixel_std_1': 72.468, 'pixel_std_2': 72.195},
'original_split_3': {
    'pixel_mean_0': 161.983, 'pixel_mean_1': 162.134, 'pixel_mean_2': 162.231,
    'pixel_std_0': 72.398, 'pixel_std_1': 72.468, 'pixel_std_2': 72.195},
'original_split_2_aug_6': {
    'pixel_mean_0': 97.594, 'pixel_mean_1': 103.687, 'pixel_mean_2': 109.030,
    'pixel_std_0': 95.714, 'pixel_std_1': 95.744, 'pixel_std_2': 95.355},
'original_split_3_aug_6': {
    'pixel_mean_0': 97.711, 'pixel_mean_1': 103.910, 'pixel_mean_2': 109.287,
    'pixel_std_0': 95.788, 'pixel_std_1': 95.909, 'pixel_std_2': 95.496},
'maps_filtered': {
    'pixel_mean_0': 128.833, 'pixel_mean_1': 128.790, 'pixel_mean_2': 128.829, 'pixel_std_0': 54.380,
    'pixel_std_1': 54.207, 'pixel_std_2': 54.045},
'maps_filtered_no_original_aug_6': {
    'pixel_mean_0': 95.398, 'pixel_mean_1': 102.010, 'pixel_mean_2': 108.387, 'pixel_std_0': 94.464, 'pixel_std_1': 94.473, 'pixel_std_2': 94.066},
'maps_filtered_aug_6': {
    'pixel_mean_0': 135.113, 'pixel_mean_1': 140.659, 'pixel_mean_2': 145.981, 'pixel_std_0': 87.435,
    'pixel_std_1': 84.976, 'pixel_std_2': 82.163}
}

# in case multiple jobs are running in the same directory, SESSION_ID will prevent name conflicts
CODEBASE_SNAPSHOT_ZIP_NAME = f"codebase_{SESSION_ID}.zip"
CHECKPOINTS_DIR = os.path.join("checkpoints", str(SESSION_ID))
LOGGING_DIR = "logs/"
COMMAND_LINE_FILE_NAME = "command_line.txt"
OUTPUT_PRED_DIR = "output_preds"
SALIENCY_MAP_DIR = "saliency_maps"
OUTPUT_FLOAT_DIR = "float_maps"
MLFLOW_FTP_USER = "mlflow_user"
MLFLOW_HTTP_USER = "cil22"
MLFLOW_HTTP_PASS = "equilibrium"
MLFLOW_HOST = "algvrithm.com"
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:8000"
MLFLOW_JUMP_HOST = "eu-login-01"
MLFLOW_FTP_PASS_URL = "https://algvrithm.com/files/mlflow_cil_pass.txt"
MLFLOW_PROFILING = False
# Pushbullet access token to use for sending notifications about critical events such as exceptions during training
# (None to avoid sending Pushbullet notifications)
DEFAULT_PUSHBULLET_ACCESS_TOKEN = pathlib.Path('pb_token.txt').read_text() if os.path.isfile('pb_token.txt') else None


###########################################################################################
##################################    helper functions    #################################
###########################################################################################

def consistent_shuffling(*args):
    """
    Randomly permutes all lists in the input arguments such that elements at the same index in all lists are still
    at the same index after the permutation.
    """
    z = list(zip(*args))
    random.shuffle(z)
    return list(map(list, zip(*z)))


def next_perfect_square(n):
    """
    Given a number n, compute the next perfect square (e.g. n=19 -> 25)
    """
    next_n = math.floor(math.sqrt(n)) + 1
    return next_n * next_n


def is_perfect_square(n):
    """
    Returns whether the given number is a perfect square
    """
    x = math.sqrt(n)
    return (x - math.floor(x)) == 0


def sftp_download_dir_portable(sftp, remote_dir, local_dir, preserve_mtime=False, callback=None):
    """
    Cross-platform SFTP directory downloading
    Code adapted from https://stackoverflow.com/a/50130813
    Args:
        sftp: pysftp connection object
        remote_dir: remote directory path
        local_dir: local directory path
        preserve_mtime: argument for the sftp.get function
        callback: callback for the sftp.get function
    """
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = remote_dir + "/" + entry.filename
        local_path = os.path.join(local_dir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode):
            try:
                os.mkdir(local_path)
            except OSError:
                pass
            sftp_download_dir_portable(sftp, remote_path, local_path, preserve_mtime, callback)
        elif S_ISREG(mode):
            sftp.get(remote_path, local_path, preserve_mtime=preserve_mtime, callback=callback)


def create_or_clean_directory(dir_name):
    """
    Create output directory if it does not exist, or clean it if it does
    Args:
        dir_name (str): Name of the directory we want to create/clean
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        for f in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, f))


def flatten(xss):
    """
    Flatten a list of lists of lists
    """
    return [x for xs in xss for x in xs]

def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
