import os
import cv2
import numpy as np
import shutil
import argparse


def new_dir(dir_name):
    if dir_name is not None:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


def split_image_gt(input_dir, output_dir):
    new_dir(output_dir)
    new_dir(os.path.join(output_dir, 'images'))
    new_dir(os.path.join(output_dir, 'groundtruth'))
    for file_name in os.listdir(input_dir):
        file_no_extension = file_name.split('.')[0]
        img_and_gt = cv2.imread(os.path.join(input_dir, file_name))
        width = img_and_gt.shape[1]
        img = img_and_gt[:, :width//2, :]
        gt = img_and_gt[:, width//2:, :]
        cv2.imwrite(os.path.join(output_dir, f'images/{file_no_extension}.jpg'), img)
        cv2.imwrite(os.path.join(output_dir, f'groundtruth/{file_no_extension}.png'), gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='Path to input directory')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Path to output directory')
    options = parser.parse_args()

    input_dir = options.input_dir  # 'to_be_removed/script_3_output'
    output_dir = options.output_dir  # 'to_be_removed/script_4_output'

    split_image_gt(input_dir=input_dir, output_dir=output_dir)

