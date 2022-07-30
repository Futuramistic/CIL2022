import os
import cv2
import shutil
import argparse


"""
Split a directory with merged image/groundtruths into 2 separate directories
"""


def new_dir(dir_name):
    """
    Force create a new directory with given name
    Args:
        dir_name (str): Name of the new directory
    """
    if dir_name is not None:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


def split_image_gt(input_dir, output_dir):
    """
    For each composite image in the input directory (consisting of an image next to its groundtruth), split it
    and output it in the output_dir
    Args:
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
    """
    # Create the output directories
    new_dir(output_dir)
    new_dir(os.path.join(output_dir, 'images'))
    new_dir(os.path.join(output_dir, 'groundtruth'))
    for file_name in os.listdir(input_dir): # Split each composite image
        file_no_extension = file_name.split('.')[0]
        img_and_gt = cv2.imread(os.path.join(input_dir, file_name))
        width = img_and_gt.shape[1]
        img = img_and_gt[:, :width//2, :]
        gt = img_and_gt[:, width//2:, :]
        cv2.imwrite(os.path.join(output_dir, f'images/{file_no_extension}.png'), img)
        cv2.imwrite(os.path.join(output_dir, f'groundtruth/{file_no_extension}.png'), gt)


if __name__ == '__main__':
    desc_str = 'Split a directory with merged image/groundtruths into 2 separate directories'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='Path to input directory')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Path to output directory')
    options = parser.parse_args()

    input_dir = options.input_dir
    output_dir = options.output_dir

    split_image_gt(input_dir=input_dir, output_dir=output_dir)

