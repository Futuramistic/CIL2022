"""
This file absolutely has to be in a subdirectory of the root directory of the project, because of the ROOT_DIR value
DO NOT MOVE!
"""

import math
import os
import random
from stat import S_ISDIR, S_ISREG
import sys
import time


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
DEFAULT_TF_INPUT_SHAPE = (None, None, 3)
DATASET_ZIP_URLS = {
    # "original": dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
    "original": "https://polybox.ethz.ch/index.php/s/x2RcSv4MOG3rtPB/download",

    # "original_128": "original" dataset, patchified into 128x128 patches and augmented using Preprocessor
    # WARNING: take into account that this dataset has 2160/720 train/"unlabeled test" images (original has only 144/144, resp.)
    # e.g. use fewer epochs and a high train fraction (high "split" value), so the evaluation doesn't take too long!
    "original_128": "https://polybox.ethz.ch/index.php/s/c68pfFLBXCCjzDT/download",

    # "original_256": "original" dataset, patchified into 256x256 patches and augmented using Preprocessor
    # WARNING: take into account that this dataset has 2160/720 train/"unlabeled test" images (original has only 144/144, resp.)
    # e.g. use fewer epochs and a high train fraction (high "split" value), so the evaluation doesn't take too long!
    "original_256": "https://polybox.ethz.ch/index.php/s/ncSp9vsJ1HAIHcR/download"
}
# in case multiple jobs are running in the same directory, SESSION_ID will prevent name conflicts
CODEBASE_SNAPSHOT_ZIP_NAME = f"codebase_{SESSION_ID}.zip"
CHECKPOINTS_DIR = os.path.join("checkpoints", str(SESSION_ID))
LOGGING_DIR = "logs/"
MLFLOW_USER = "mlflow_user"
MLFLOW_HOST = "algvrithm.com"
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:8000"
MLFLOW_JUMP_HOST = "eu-login-01"
MLFLOW_PASS_URL = "https://algvrithm.com/files/mlflow_cil_pass.txt"
MLFLOW_PROFILING = False


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
    next_n = math.floor(math.sqrt(n)) + 1
    return next_n * next_n


def is_perfect_square(n):
    x = math.sqrt(n)
    return (x - math.floor(x)) == 0


# Cross-platform SFTP directory downloading code adapted from https://stackoverflow.com/a/50130813
def sftp_download_dir_portable(sftp, remote_dir, local_dir, preserve_mtime=False):
    # sftp: pysftp connection object
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = remote_dir + "/" + entry.filename
        local_path = os.path.join(local_dir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode):
            try:
                os.mkdir(local_path)
            except OSError:     
                pass
            sftp_download_dir_portable(sftp, remote_path, local_path, preserve_mtime)
        elif S_ISREG(mode):
            sftp.get(remote_path, local_path, preserve_mtime=preserve_mtime)
