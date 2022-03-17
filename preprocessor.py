import os
import argparse
from data_handling import torchDataset
from utils import ROOT_DIR
import warnings
import shutil


def main():
    # Define the parser
    parser = argparse.ArgumentParser(description='Implementation of Preprocessor that augments and segments the '
                                                 'dataset into patches')
    parser.add_argument('-d', '--dataset', type=str, default='original')
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-a', '--augmentation_factor', type=int, default=5)
    # Parse
    params = parser.parse_args()

    # Retrieve the parameters
    dataset = params.dataset
    output_dir = params.output_dir
    augmentation_factor = params.augmentation_factor

    # Check whether the directories are correctly specified
    if not os.path.exists(f'{ROOT_DIR}/dataset/{dataset}'):
        print(f"The specified dataset at {ROOT_DIR}/dataset/{dataset} does not exist.")
        exit(-1)
    if output_dir is None:
        print(f"Please specify a name for the output directory with parameter '-o' or '--output'")
        exit(-1)

    # set input data directories
    img_path = f'dataset/{dataset}/training/images'
    gt_path = f'dataset/{dataset}/training/groundtruth'

    # check whether the output directory does not already exist
    output_path = f'{ROOT_DIR}/dataset/{output_dir}'
    if os.path.exists(output_path):
        retry = True
        while retry:
            answer = input("The specified output directory already exists. Do you want to overwrite it? [y/n]")
            retry = False
            if answer == 'y':
                print(f'Overwriting {output_path}')
                shutil.rmtree(output_path)
            elif answer == 'n':
                new_output_dir = input('Please specify another output directory: ')
                if os.path.exists(f'{ROOT_DIR}/dataset/{new_output_dir}'):
                    retry = True
                else:
                    output_dir = new_output_dir
            else:
                print("Please answer with 'y' or 'n'")
                retry = True

    # set output data directories
    output_path = f'{ROOT_DIR}/dataset/{output_dir}'
    output_img_path = f'dataset/{output_dir}/training/images'
    output_gt_path = f'dataset/{output_dir}/training/groundtruth'

    if create_file_structure(output_path) == -1:
        print("Aborting")
        exit(-1)

    print("Created output file structure")

    # read the input data
    original_dataset = torchDataset.SegmentationDataset(img_path, gt_path)
    nb_images = original_dataset.__len__()
    for i in range(nb_images):
        image, gt = original_dataset.__getitem__(i)
        print(f'image shape {image.shape}')
        print(f'gt shape {gt.shape}')
        break

    print(f"output dir {output_dir}")



def create_file_structure(destination_path):
    # create paths
    try:
        os.makedirs(destination_path)
        os.makedirs(f"{destination_path}/training")
        os.makedirs(f"{destination_path}/training/images")
        os.makedirs(f"{destination_path}/training/groundtruth")
        os.makedirs(f"{destination_path}/test")
        os.makedirs(f"{destination_path}/test/images")
        return 1
    except:
        warnings.warn(
            f"Ooops, there has been an error while creating the folder structure in {destination_path} in the preprocessor")
        return -1


if __name__ == "__main__":
    main()
