import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


"""
Compute the statistics of a given dataset (mean and standard deviation for each channel)
"""


def main(dataset):
    ds_images_dir = os.path.join('dataset', dataset, 'training', 'images')
    if not os.path.isdir(ds_images_dir):
        print(f'Directory "{ds_images_dir}" (supposed to contain images from which to compute '
              f'dataset stats) does not exist')
        return
    
    png_filenames = []
    for filename in os.listdir(ds_images_dir):
        if filename.lower().endswith('.png'):
            png_filenames.append(filename)

    if len(png_filenames) == 0:
        print(f'No PNG files found in directory "{ds_images_dir}"')
        return

    num_pixels = 0

    pixel_sum_0 = 0
    pixel_sum_1 = 0
    pixel_sum_2 = 0

    # calculate means
    for filename in tqdm(png_filenames):
        with Image.open(os.path.join(ds_images_dir, filename)) as img:
            arr = np.asarray(img)
            pixel_sum_0 += arr[0].sum()
            pixel_sum_1 += arr[1].sum()
            pixel_sum_2 += arr[2].sum()
            num_pixels += arr.shape[-2] * arr.shape[-1]
    
    pixel_mean_0 = pixel_sum_0 / num_pixels
    pixel_mean_1 = pixel_sum_1 / num_pixels
    pixel_mean_2 = pixel_sum_2 / num_pixels
    
    pixel_sq_sum_0 = 0
    pixel_sq_sum_1 = 0
    pixel_sq_sum_2 = 0

    # calculate stds
    for filename in png_filenames:
        with Image.open(os.path.join(ds_images_dir, filename)) as img:
            arr = np.asarray(img)
            pixel_sq_sum_0 += ((arr[0] - pixel_mean_0) ** 2.0).sum()
            pixel_sq_sum_1 += ((arr[1] - pixel_mean_1) ** 2.0).sum()
            pixel_sq_sum_2 += ((arr[2] - pixel_mean_2) ** 2.0).sum()
    
    pixel_std_0 = np.sqrt(pixel_sq_sum_0 / num_pixels)
    pixel_std_1 = np.sqrt(pixel_sq_sum_1 / num_pixels)
    pixel_std_2 = np.sqrt(pixel_sq_sum_2 / num_pixels)

    vals = {'pixel_mean_0': pixel_mean_0, 'pixel_mean_1': pixel_mean_1, 'pixel_mean_2': pixel_mean_2,
            'pixel_std_0': pixel_std_0, 'pixel_std_1': pixel_std_1, 'pixel_std_2': pixel_std_2}

    print(f'Stats for "{dataset}" dataset: {vals}')


if __name__ == '__main__':
    desc_str = 'Compute the statistics of a given dataset (mean and standard deviation for each channel)'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset to compute stats for')
    options = parser.parse_args()
    main(options.dataset)
