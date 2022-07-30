import argparse
import os
import shutil
from tqdm import tqdm
import re


"""
Create sub-datasets with different train-validation splits from the given dataset
"""


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    # from https://stackoverflow.com/a/5967539
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def main(dataset, num_splits, val_samples_per_split):
    ds_images_dir = os.path.join('dataset', dataset, 'training', 'images')
    ds_gt_dir = os.path.join('dataset', dataset, 'training', 'groundtruth')

    if not os.path.exists(ds_images_dir):
        print(f'Dataset "{dataset}" not found; please run "dataset_downloader.py" first')
        return

    png_filenames = []
    for filename in sorted(os.listdir(ds_images_dir), key=natural_keys):
        if filename.lower().endswith('.png'):
            png_filenames.append(filename)

    if len(png_filenames) == 0:
        print(f'No PNG files found in directory "{ds_images_dir}"')
        return

    split_ds_val_start_idx = 0
    for split_idx in tqdm(range(num_splits)):
        split_ds_dir = os.path.join('dataset', dataset + f'_split_{split_idx+1}')

        ds_val_filenames = png_filenames[split_ds_val_start_idx:split_ds_val_start_idx+val_samples_per_split]

        if os.path.exists(split_ds_dir):
            retry = True
            while retry:
                answer = input(f'The output directory "{split_ds_dir}" already exists. '
                               f'Do you want to overwrite it? [y/n]')
                retry = False
                if answer == 'y':
                    print(f'Overwriting "{split_ds_dir}"')
                    shutil.rmtree(split_ds_dir)
                elif answer == 'n':
                    new_output_dir = input('Please specify another output directory: ')
                    if os.path.exists(os.path.join('dataset', new_output_dir)):
                        retry = True
                    else:
                        split_ds_dir = os.path.join('dataset', new_output_dir)
                else:
                    print('Please answer with "y" or "n"')
                    retry = True

        os.makedirs(os.path.join(split_ds_dir, 'training'))
        shutil.copytree(os.path.join('dataset', dataset, 'test'), (os.path.join(split_ds_dir, 'test')))
        os.makedirs(os.path.join(split_ds_dir, 'training', 'images'))
        os.makedirs(os.path.join(split_ds_dir, 'training', 'groundtruth'))

        sequential_sample_idx = 0
        for sample_idx, png_filename in tqdm(enumerate(png_filenames)):
            val_sample_idx = -1 if png_filename not in ds_val_filenames else ds_val_filenames.index(png_filename)
            is_val_sample = val_sample_idx > -1
            new_filename = f'val_sample_{val_sample_idx}.png' if is_val_sample\
                           else f'sample_{sequential_sample_idx}.png'
            
            shutil.copyfile(os.path.join(ds_images_dir, png_filename),
                            os.path.join(split_ds_dir, 'training', 'images', new_filename))
            
            shutil.copyfile(os.path.join(ds_gt_dir, png_filename),
                            os.path.join(split_ds_dir, 'training', 'groundtruth', new_filename))

            if not is_val_sample:
                sequential_sample_idx += 1

        split_ds_val_start_idx += len(ds_val_filenames)
    
    print(f'Created {num_splits} splits of the "{dataset}" dataset')


if __name__ == '__main__':
    desc_str = 'Create sub-datasets with different train-validation splits from the given dataset'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset to create splits for')
    parser.add_argument('-n', '--num_splits', required=True, type=int, help='Number of splits to create')
    parser.add_argument('-v', '--val_samples_per_split', required=True, type=int,
                        help='Number of validation samples to use per split')
    options = parser.parse_args()

    dataset = options.dataset
    num_splits = options.num_splits
    val_samples_per_split = options.val_samples_per_split

    main(dataset, num_splits, val_samples_per_split)
