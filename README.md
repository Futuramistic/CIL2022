# CIL Course Project: Road Segmentation

This repository contains the implementation of our solution for the Road Segmentation task of the Computational Intelligence Lab.

Please refer to the accompanying [paper](https://arxiv.org/abs/2030.12345) for an in-depth description of our 
research process and results.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Authors
* Alexey Gavryushin
* Mateusz Nowak
* Anne Marx
* Noureddine Gueddach

## Code structure
>📋  TODO: put the code structure here

## Requirements

The implementation works both on Linux and Windows.

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

#### Available datasets
For reproducing our results, you do not need to setup a dataset folder, you 
just need to specify the dataset name in the command line when training and 
our framework will automatically download the relevant dataset. Available 
dataset names are:

>📋  TODO: Only keep the datasets that we need

> * original: dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
> * ext_original: "original" dataset, extended with 80 images scraped from Google Maps
> * new_original: "original" dataset, with first 25 samples moved to end to form the validation split
> * new_ext_original: "ext_original" dataset, with first 25 samples moved to end to form the validation split
> * new_ext_original_oversampled: "ext_original" dataset, with second city class oversampled, and 
first 25 sample moved to end to form the validation split
> * original_gt: dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition, 
>but with images replaced by ground truth
> * original_128: "original" dataset, patchified into 128x128 patches and augmented using Preprocessor
> * original_256: "original" dataset, patchified into 256x256 patches and augmented using Preprocessor

    # "additional_maps_1": dataset retrieved from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/ -> maps.tar.gz
    # and processed to convert the RGB masks to B/W. The resulting masks are not perfect and could definitely
    # use some better processing, but they seem to be good enough visually
    "new_maps": "https://polybox.ethz.ch/index.php/s/QTxb24YMpL1Rs66/download",

    # Massachusetts Road dataset (256x256)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 25,328 Images
    # Testing: 1,293 Images
    # WARNING: some images have white or black outer values due to processing (mostly bottom or right side)
    "massachusetts_256":"https://polybox.ethz.ch/index.php/s/WnctKQV89H6W7KT/download",

    # Massachusetts Road dataset (128x128)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 81,669 Images
    # Testing: 4,176 Images
    # WARNING: some images have white or black outer values due to processing (mostly bottom or right side)
    "massachusetts_128":"https://polybox.ethz.ch/index.php/s/XjSto2pXCeZydiH/download",

    # Massachusetts Road dataset (400x400) + the original (400x400)
    # WARNING: EXTRA LARGE dataset (!!!) 
    # Training: 12,982 Images - Massachusetts (testing+training) + original trainig
    # Testing: 144 Images - only original testing images
    # WARNING: Some images have white or black outer values due to processing (mostly bottom or right side)!
    #          However, the number of "partial" images is limited
    "large":"https://polybox.ethz.ch/index.php/s/uXJgQrQazhrn5gA/download",

    # "original_aug_6": "original" dataset, 400x400 but with augmented training set using Preprocessor (x6)
    # I usually use a 0.975 split for this dataset
    "original_aug_6": "https://polybox.ethz.ch/index.php/s/ICjaUr4ayCNwySJ/download",

    # Recreation of "original_aug_6" dataset, but with 25 samples from original dataset excluded from augmentation
    # procedure to avoid data leakage; same 25 samples as in "new_original", "new_ext_original" and "ext_original_aug_6" datasets
    # use split of 0.971 to use exactly these 25 samples as the validation set
    "new_original_aug_6": "https://polybox.ethz.ch/index.php/s/LJZ0InoG6GwyGsC/download",

    # Recreation of "original_aug_6" dataset, but with 80 additional samples scraped from Google Maps added before
    # augmentation procedure, and with 25 samples from original dataset excluded from augmentation procedure
    # to avoid data leakage; same 25 samples as in "new_original", "new_ext_original" and "new_original_aug_6" datasets
    # use split of 0.9825 to use exactly these 25 samples as the validation set
    "ext_original_aug_6": "https://polybox.ethz.ch/index.php/s/9hDXLlX7mB5Xljq/download",
    
    # Recreation of "original_aug_6" dataset, but with 80 additional samples scraped from Google Maps added before
    # augmentation procedure, the second city class oversampled, and with 25 samples from original dataset excluded from
    # augmentation procedure to avoid data leakage; same 25 samples as in "new_original", "new_ext_original" and
    # "new_original_aug_6" datasets; use split of 0.9875 to use exactly these 25 samples as the validation set
    "ext_original_aug_6_oversampled": "https://polybox.ethz.ch/index.php/s/9hDXLlX7mB5Xljq/download"

#### Custom datasets
If you want to setup a custom dataset, simply place it in the 'dataset' folder,
with the following structure:

```
.
└── dataset                   
    └── <dataset-name>        
        └── training
            ├── groundtruth
            └── images
        └── test
            └── images
```

## Models
We trained many different models, all of which are available in the `models` folder.
The available models are (with the respective name to pass to the training
command-line below under the `model-name` argument):

*Tensorflow models*
```
* UNet (name: 'unettf')
* UNet++ (name: 'unet++')
* UNetExp (name: 'unetexp')
* U2Net (name: 'u2net')
* Attention UNet ('attunet')
* Attention UNet++ ('attunet++')
* GL Dense UNet ('gldenseunet')

```

*Torch models*
```
* UNet (Torch implementation - name: 'unet')
* CRA-Net (name: 'cranet')
* DeepLabV3 (name: 'deeplabv3')
* Fast SCNN (name: 'fastscnn')
```

>📋  TODO: Only keep the models that have been tested

*Reinforcement Learning models (Torch)*
>📋  TODO: How should we call our RL model?

## Training

To train our models in the paper, run this command:

```train
python main.py -d <dataset-name> -m <model-name> -E <experiment-name> 
--batch_size <batch-size> --num_epochs <num-epochs>
```

For example:

```train
python main.py -d original -m deeplabv3 -E training_run 
--batch_size 4 --num_epochs 30
```

>📋  TODO put commands for our final models

## Evaluation

When a model is trained, checkpoints appear in the `checkpoints` folder.
To evaluate a model, run:

```eval
python torch_predictor.py -m <model-name> -c <checkpoint-name>
```

if it's a Torch model or:

```eval
python tf_predictor.py -m <model-name> -c <checkpoint-name>
```

if it's a Tensorflow model.

The predictions appear in the `output_preds` directory

## Pre-trained Models

>📋 TODO should we make available some pretrained models?

You can download pretrained models here:

- [Pretrained model name 1](https://drive.google.com/mymodel.pth) trained on 'blabla' using parameters 'blabla'. 

## Results

Our models achieve the following performance on the 'original' dataset:

| Model name         | F1-Score  |
| ------------------ |---------------- |
| Our RL model name  |     0.999         |
| Our ensemble  |     0.998         |
| Other?  |     0.997         |

>📋  Our model '<insert-name>' Achieved an F1-Score of <insert-score> on the Kaggle competition, granting us 
>the `x'th` position on the Leaderboard. 
