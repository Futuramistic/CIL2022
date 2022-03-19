import os
import argparse
from data_handling import dataloader, torchDataset
from utils import ROOT_DIR
import warnings
import shutil
from torchvision.utils import save_image
import torch
from torchvision import transforms
from tqdm import tqdm


# taken from https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = 'argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def main():
    # Define the parser
    parser = argparse.ArgumentParser(description='Implementation of Preprocessor that augments and segments the '
                                                 'dataset into patches')
    parser.add_argument('-d', '--dataset', type=str, default='original')
    parser.add_argument('-o', '--output_dir', type=str)

    # Augmentation parameters
    parser.add_argument('-a', '--augmentation_factor', type=int, default=3)
    parser.add_argument('--rotation', type=int, default=10)
    parser.add_argument('--translation', type=float, nargs='+', action=required_length(2, 2), default=[0.1, 0.1])
    parser.add_argument('--scale', type=float, nargs=2, action=required_length(2, 2), default=[0.8, 1.2])

    parser.add_argument('--vertical_flip', dest='vertical_flip', action='store_true')
    parser.add_argument('--no-vertical_flip', dest='vertical_flip', action='store_false')
    parser.set_defaults(vertical_flip=True)
    parser.add_argument('--horizontal_flip', dest='horizontal_flip', action='store_true')
    parser.add_argument('--no-horizontal_flip', dest='horizontal_flip', action='store_false')
    parser.set_defaults(horizontal_flip=True)

    # Whether to create smaller patches
    parser.add_argument('--patchify', type=int, default=0)

    # TODO what are good ranges for these parameters?
    parser.add_argument('--brightness', type=float, nargs=2, action=required_length(2, 2), default=[0.8, 1.2])
    parser.add_argument('--contrast', type=float, nargs=2, action=required_length(2, 2), default=[0.8, 1.2])
    parser.add_argument('--saturation', type=float, nargs=2, action=required_length(2, 2), default=[0.8, 1.2])

    # Parse
    try:
        params = parser.parse_args()
    except argparse.ArgumentTypeError as err:
        print(err)
        exit(-1)

    # Retrieve the parameters
    dataset = params.dataset
    output_dir = params.output_dir
    augmentation_factor = params.augmentation_factor
    rotation = params.rotation
    translation = tuple(params.translation)
    scale = tuple(params.scale)
    brightness = params.brightness
    contrast = params.contrast
    saturation = params.saturation
    vertical_flip = params.vertical_flip
    horizontal_flip = params.horizontal_flip
    patch_size = params.patchify
    patchify = patch_size > 0

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
            answer = input("The specified output directory already exists. Do you want to overwrite it? [y/n] ")
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

    if __create_file_structure(output_path, patchify) == -1:
        print("Aborting")
        exit(-1)
    print("Created output file structure")

    # Create the transformations
    patching_transform = transforms.FiveCrop(patch_size) if patchify else None  # Check we want patches or not
    geometric_transforms = [transforms.RandomAffine(degrees=rotation, translate=translation, scale=scale)]
    if vertical_flip:
        geometric_transforms.append(transforms.RandomVerticalFlip())
    if horizontal_flip:
        geometric_transforms.append(transforms.RandomHorizontalFlip())
    augmentation_parameters = {
        'factor': augmentation_factor,
        'patching_transform': patching_transform,
        'geometric_transforms': geometric_transforms,
        'lighting_transforms': [
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
        ]
    }

    # read and process the dataset
    img_paths, gt_paths = dataloader.DataLoader.get_img_gt_paths(img_path, gt_path)
    original_dataset = torchDataset.SegmentationDataset(img_paths, gt_paths)
    __process_dataset(original_dataset, output_img_path, output_gt_path, augmentation_parameters)

    input_test_dir = f'dataset/{dataset}/test/images'
    output_test_dir = f'dataset/{output_dir}/test/images'
    input_test_path = f'{ROOT_DIR}/{input_test_dir}'
    output_test_path = f'{ROOT_DIR}/{output_test_dir}'
    if not patchify:
        # copy the test files to the new dataset without processing
        shutil.copytree(input_test_path, output_test_path)
    else:
        # patchify the test set too
        input_test_paths, _ = dataloader.DataLoader.get_img_gt_paths(input_test_dir, None)
        original_test_dataset = torchDataset.SegmentationDataset(input_test_paths)
        __patchify_test_set(original_test_dataset, output_test_dir, patching_transform)
        pass

    print(f'Finished creating the new dataset at {output_path}')


def __patchify_test_set(dataset, output_test_dir, patchify_function):
    torch.manual_seed(42)
    nb_images = dataset.__len__()

    # Transform the images
    print('Patchifying test set...')
    output_size = 0
    for i in tqdm(range(nb_images)):
        image = dataset.__getitem__(i)
        image = image / 255.
        patches = list(patchify_function(image))
        for k, patch in enumerate(patches):
            # save the image
            save_image(patch, f'{output_test_dir}/image_{i}_{k + 1}.png')
            output_size += 1
    print(f'Created output test set with {output_size} images')


def __process_dataset(dataset, output_img_path, output_gt_path, augmentation_parameters):
    torch.manual_seed(42)
    nb_images = dataset.__len__()

    # Define transformations
    patching_function = augmentation_parameters['patching_transform']  # (dataset.shape[1])
    patching_transform = transforms.Compose([
        patching_function,
        transforms.Lambda(lambda crops: list(crops))
    ]) if patching_function is not None else None
    geometric_transforms = torch.nn.Sequential(
        *augmentation_parameters['geometric_transforms']
    )
    lighting_transforms = torch.nn.Sequential(
        *augmentation_parameters['lighting_transforms']
    )
    scripted_geometric_transforms = torch.jit.script(geometric_transforms)
    scripted_lighting_transforms = torch.jit.script(lighting_transforms)

    # Transform the images
    print('Transforming images...')
    output_size = 0
    for i in tqdm(range(nb_images)):
        image, gt = dataset.__getitem__(i)
        image = image / 255.  # transform from ByteTensor to FloatTensor
        gt = gt / 255.  # transform from ByteTensor to FloatTensor

        # TODO: this assumes RGBA format. Must handle RGB and grayscale formats as well
        concatenated = torch.cat((gt, image), dim=0)

        if patching_function is None:
            patches = torch.unsqueeze(concatenated, 0)
            # save original images
            save_image(gt, f'{output_gt_path}/image_{i}_0.png')
            save_image(image, f'{output_img_path}/image_{i}_0.png')
        else:
            patches = patching_transform(concatenated)

        for k, patch in enumerate(patches):
            for j in range(augmentation_parameters['factor']):
                # transform the image
                transformed = scripted_geometric_transforms(patch)  # apply to both image and ground_truth
                tr_gt, tr_image, alpha = torch.split(transformed, [1, transformed.shape[0] - 2, 1], dim=0)
                tr_image = scripted_lighting_transforms(tr_image)  # only apply to rgb channels of image
                tr_image = torch.cat((tr_image, alpha), dim=0)
                # save the image
                save_image(tr_gt, f'{output_gt_path}/image_{i}_{k+1}_{j + 1}.png')
                save_image(tr_image, f'{output_img_path}/image_{i}_{k+1}_{j + 1}.png')
                output_size += 1

    print(f'Created output dataset with {output_size} images')


def __create_file_structure(destination_path, create_test_dir):
    # create paths
    try:
        os.makedirs(destination_path)
        os.makedirs(f"{destination_path}/training/images")
        os.makedirs(f"{destination_path}/training/groundtruth")
        if create_test_dir:
            os.makedirs(f"{destination_path}/test/images")
        return 1
    except:
        warnings.warn(
            f"Ooops, there has been an error while creating the folder structure in {destination_path} in the preprocessor")
        return -1


if __name__ == "__main__":
    main()
