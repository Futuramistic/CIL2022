'''
This file absolutely has to be in the root directory of the project, because of the ROOT_DIR Value
DO NOT MOVE!
'''

import os
# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif"]
DATASET_ZIP_URLS = {
    "original": "https://polybox.ethz.ch/index.php/s/x2RcSv4MOG3rtPB/download"
}
