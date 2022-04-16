"""
This file absolutely has to be in a subdirectory of the root directory of the project, because of the ROOT_DIR value
DO NOT MOVE!
"""

import math
import os
import random

###########################################################################################
##################################    global constants    #################################
###########################################################################################

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go one dir up from the dir this file is in
ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif"]
DEFAULT_SEGMENTATION_THRESHOLD = 0.5
DEFAULT_TRAIN_FRACTION = 0.8
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
CODEBASE_SNAPSHOT_ZIP_NAME = "codebase_snapshot.zip"
CHECKPOINTS_DIR = "checkpoints/"
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
