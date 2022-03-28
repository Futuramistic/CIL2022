'''
This file absolutely has to be in the root directory of the project, because of the ROOT_DIR Value
DO NOT MOVE!
'''

import os
import random
import math
from models.learning_aerial_image_segmenation_from_online_maps import Deeplabv3, Unet, Fastscnn
from models.road_extraction_from_high_res_rsi_using_dl.gl_dense_u_net import *


###########################################################################################
##################################    global constants    #################################
###########################################################################################

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif"]
MODEL_CLASS_DICT = {'deeplabv3': Deeplabv3.Deeplabv3,
                    'fastscnn': Fastscnn.FastSCNN,
                    'unet': Unet.UNet}
DEFAULT_TRAIN_FRACTION = 0.8
DATASET_ZIP_URLS = {
    # "original": dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
    "original": "https://polybox.ethz.ch/index.php/s/x2RcSv4MOG3rtPB/download",

    # "original_128": "original" dataset, patchified into 128x128 patches and augmented using Preprocessor
    "original_128": "https://polybox.ethz.ch/index.php/s/c68pfFLBXCCjzDT/download",

    # "original_256": "original" dataset, patchified into 256x256 patches and augmented using Preprocessor
    "original_256": "https://polybox.ethz.ch/index.php/s/ncSp9vsJ1HAIHcR/download"
}
CODEBASE_SNAPSHOT_ZIP_NAME = "codebase_snapshot.zip"
MLFLOW_USER = "mlflow_user"
MLFLOW_HOST = "algvrithm.com"
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:8000"
MLFLOW_PASS_URL = "https://algvrithm.com/files/mlflow_cil_pass.txt"
MLFLOW_PROFILING = True


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
