# CIL Course Project: Road Segmentation

This repository contains the implementation of our solution for the Road Segmentation task of the Computational Intelligence Lab.

Please refer to the accompanying [<insert_link_to_pdf>](https://arxiv.org/abs/2030.12345) for an in-depth description of our 
research process and results.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Authors
* Alexey Gavryushin
* Mateusz Nowak
* Anne Marx
* Noureddine Gueddach

## Our image segmentation Framework
We conceived our framework to be able to handle both tensorflow-based models and torch-based models, as both are ubiquitous in the literature. To track our experiments, we used the MLFlow framework. Although it makes the code base a bit more complex, it allowed us to have a centralized space where we could log and **share** our experiments, model checkpoints, hyperparameters and so on. For hyperparameter tuning, we use [HyperOpt](http://hyperopt.github.io/hyperopt/), a framework conceived to optimize hyper parameter searches.

The overall structure of the repository is described below.

## Code structure

```
.
├── data_handling                   # Scripts for loading datasets
└── dataset                         # Folder containing the datasets
    ├── <dataset-name-1>
    ├── <dataset-name-2>  
    └── ...     
├── factory                         # Helper for Loading models
├── hyperopt_                       # Utilities for hyperparameter searches (using the HyperOpt framework)
├── losses                          # Folder with losses used across models
└── models                          # Folder with our various models
    ├── reinforcement               # RL models
    ├── torch                       # Torch models
    └── TF                          # Tensorflow models
├── trainers                        # Custom trainers for each model
├── utils                           # Utility functions and constants
└── processing                      # Contains processing files such as preprocessing, downloading etc.
torch_predictor.py                  # Script for making predictions on a torch model
tf_predictor.py                     # Script for making predictions on a tf model
mask_to_submission.py               # Script from the Kaggle competition page
submission_to_mask.py               # Script from the Kaggle competition page
main_hyperopt.py                    # Script for launching HyperOpt experiments
main.py                             # Script for training models
...
```

## Requirements

The implementation works both on Linux and Windows.

To setup an environment, run:

```setup
conda create -n CIL2022 python==3.7
conda activate CIL2022
conda install cmake
```

To install the requirements, run:

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
but with images replaced by ground truth
> * original_128: "original" dataset, patchified into 128x128 patches and augmented using Preprocessor
> * original_256: "original" dataset, patchified into 256x256 patches and augmented using Preprocessor
> * additional_maps_1: dataset retrieved from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
> * massachusetts_400: Massachusetts Road dataset (400x400)
> * massachusetts_256: Massachusetts Road dataset (256x256)
> * massachusetts_128: Massachusetts Road dataset (128x128)
> * original_aug_6: "original" dataset, 400x400 but with augmented training set using Preprocessor (x6)
> * new_original_aug_6: Recreation of "original_aug_6" dataset, but with 25 samples from original dataset excluded from augmentation
procedure to avoid data leakage
> * ext_original_aug_6: Recreation of "original_aug_6" dataset, but with 80 additional samples scraped from Google Maps added before
augmentation procedure, and with 25 samples from original dataset excluded from augmentation procedure 
to avoid data leakage
> * new_original_aug_6: Recreation of "original_aug_6" dataset, but with 80 additional samples scraped from Google Maps added before
augmentation procedure, the second city class oversampled, and with 25 samples from original dataset excluded from
augmentation procedure to avoid data leakage

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
* UNet3+ (name: 'unet3+')
* Attention UNet ('attunet')
* Attention UNet++ ('attunet++')
* GL Dense UNet ('gldenseunet')
```

*Torch models*
```
* UNet (name: 'unet')
* CRA-Net (name: 'cranet')
* DeepLabV3 (name: 'deeplabv3')
* DeepLabV3+GAN (name: 'deeplabv3plusgan')
* SegFormer (name: 'segformer')
* TwoShotNet (name: 'twoshotnet')
```

>📋  TODO: Only keep the models that have been tested

*Reinforcement Learning models (Torch)*
```
* SimpleRLCNN (name: 'simplerlcnn')
* SimpleRLCNNMinimal (name: 'simplerlcnnminimal')
```

## Training

To train the available models, run this command:

```train
python main.py -d <dataset-name> -m <model-name> -E <experiment-name> 
--batch_size <batch-size> --num_epochs <num-epochs>
```

For example:

```train
python main.py -d original -m deeplabv3 -E training_run 
--batch_size 4 --num_epochs 30
```

>📋  TODO put commands for our final models/RL models

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
