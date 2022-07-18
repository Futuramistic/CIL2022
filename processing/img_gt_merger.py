import os
import cv2
import numpy as np
import shutil
import argparse


"""
Merge an image directory with the corresponding groundtruths directory
"""


def clean_gt(gt_dir):
    """
    Take-in a groundtruth image with shady areas and filter them out, keeping only the pure white/pure black areas
    Args:
        gt_dir (str): Path to input directory
    """
    clean_images = []
    for file_name in os.listdir(gt_dir):
        image = cv2.imread(os.path.join(gt_dir, file_name))
        dims = (image.shape[0], image.shape[1])
        small_img = cv2.resize(image, (dims[0]-1, dims[1]-1), interpolation=cv2.INTER_LINEAR)
        binary = (small_img == 255) * 255
        clean_img = cv2.resize(binary, dims, interpolation=cv2.INTER_NEAREST_EXACT)
        clean_images.append((file_name, clean_img))
    return clean_images


def filter_on_ratio(groundtruths, low=0.0, high=1.0):
    """
    Only keep images whose groundtruth white to black ratio is between 'low' and 'high'
    Args:
        groundtruths (list): List of groundtruth images
        low (float): Minimum allowed white to black ratio
        high (float): Maximum allowed white to black ratio
    """
    satisfy_ratio_images = []
    for name, image in groundtruths:
        ratio = float(np.sum(image)) / (np.prod(image.shape) * 255)
        if low < ratio < high:
            satisfy_ratio_images.append((name, image))
    return satisfy_ratio_images


def merge_image_gt(groundtruths, images_dir, output_dir):
    """
    Given the groundtruths and the images directories, merge them into a single image side by side
    Args:
        groundtruths (list): List of groundtruth images
        images_dir (str): Path to the images directory
        output_dir (str): Path to the output directory
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    img_extension = os.listdir(images_dir)[0].split('.')[-1]
    for name, gt in groundtruths:
        file_no_extension = name.split('.')[0]
        img = cv2.imread(os.path.join(images_dir, f'{file_no_extension}.{img_extension}'))
        combined = np.concatenate((img, gt), axis=1)
        cv2.imwrite(os.path.join(output_dir, name), combined)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', required=True, type=str, help='Path to the root '
                                                                             'directory of the dataset')
    parser.add_argument('--ratio_low', default=0.0, type=float, help='Lower bound on accepted white/black ratio')
    parser.add_argument('--ratio_high', default=1.0, type=float, help='Upper bound on accepted white/black ratio')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Path to output directory')
    options = parser.parse_args()

    dataset_dir = options.dataset_dir
    ratio_low = options.ratio_low
    ratio_high = options.ratio_high
    output_dir = options.output_dir

    clean_groundtruths = clean_gt(gt_dir=f'{dataset_dir}/groundtruth')
    filtered_groundtruths = filter_on_ratio(groundtruths=clean_groundtruths, low=ratio_low, high=ratio_high)
    merge_image_gt(groundtruths=filtered_groundtruths, images_dir=os.path.join(dataset_dir, 'images'),
                   output_dir=output_dir)
