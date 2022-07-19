import argparse
from data_handling import DataLoader
from utils import *

"""
Download a given dataset, using the list of known dataset URLs in "utils.py"
"""


def main(dataset):
    if DataLoader._download_data(dataset_name=dataset) == 1:
        print(f'Dataset "{dataset}" successfully downloaded or already available')
    else:
        print(f'An error occurred attempting to download the "{dataset}" dataset; see error message above')


if __name__ == '__main__':
    desc_str = 'Download a given dataset, using the list of known dataset URLs in "utils.py"'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset to download')
    options = parser.parse_args()

    main(options.dataset)
