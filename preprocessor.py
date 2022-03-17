import os
import argparse
from data_handling import torchDataset
from utils import ROOT_DIR
import warnings
import shutil
from torchvision.utils import save_image
import torch
from torchvision import transforms
from tqdm import tqdm


def main():
    # Define the parser
    parser = argparse.ArgumentParser(description='Implementation of Preprocessor that augments and segments the '
                                                 'dataset into patches')
    parser.add_argument('-d', '--dataset', type=str, default='original')
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-a', '--augmentation_factor', type=int, default=3)
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

    # read and process the dataset
    original_dataset = torchDataset.SegmentationDataset(img_path, gt_path)
    process_dataset(original_dataset, output_img_path, output_gt_path, augmentation_factor)

    # copy the test files to the new dataset without processing
    input_test_path = f'{ROOT_DIR}/dataset/{dataset}/test/images'
    output_test_path = f'{ROOT_DIR}/dataset/{output_dir}/test/images'
    shutil.copytree(input_test_path, output_test_path)

    print(f'Finished creating the new dataset at {output_path}')


def process_dataset(dataset, output_img_path, output_gt_path, augmentation_factor):
    torch.manual_seed(42)
    nb_images = dataset.__len__()

    # Define transformations
    geometric_transforms = torch.nn.Sequential(
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    lighting_transforms = torch.nn.Sequential(
        # TODO what are good ranges for these parameters?
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
    )
    scripted_geometric_transforms = torch.jit.script(geometric_transforms)
    scripted_lighting_transforms = torch.jit.script(lighting_transforms)

    # Transform the images
    print('Transforming images...')
    for i in tqdm(range(nb_images)):
        image, gt = dataset.__getitem__(i)
        image = image / 255.  # transform from ByteTensor to FloatTensor
        gt = gt / 255.  # transform from ByteTensor to FloatTensor
        # save original images
        save_image(gt, f'{output_gt_path}/image_{i}_0.png')
        save_image(image, f'{output_img_path}/image_{i}_0.png')

        # TODO: this assumes RGBA format. Must handle RGB and grayscale formats as well
        concatenated = torch.cat((gt, image), dim=0)
        for j in range(augmentation_factor):
            # transform the image
            transformed = scripted_geometric_transforms(concatenated)  # apply to both image and ground_truth
            tr_gt, tr_image, alpha = torch.split(transformed, [1, transformed.shape[0] - 2, 1], dim=0)
            tr_image = scripted_lighting_transforms(tr_image)  # only apply to rgb channels of image
            tr_image = torch.cat((tr_image, alpha), dim=0)
            # save the image
            save_image(tr_gt, f'{output_gt_path}/image_{i}_{j + 1}.png')
            save_image(tr_image, f'{output_img_path}/image_{i}_{j + 1}.png')
        break


def create_file_structure(destination_path):
    # create paths
    try:
        os.makedirs(destination_path)
        os.makedirs(f"{destination_path}/training")
        os.makedirs(f"{destination_path}/training/images")
        os.makedirs(f"{destination_path}/training/groundtruth")
        # os.makedirs(f"{destination_path}/test")
        # os.makedirs(f"{destination_path}/test/images")
        return 1
    except:
        warnings.warn(
            f"Ooops, there has been an error while creating the folder structure in {destination_path} in the preprocessor")
        return -1


if __name__ == "__main__":
    main()
