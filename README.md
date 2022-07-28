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

## Contents
1. [Our Image Segmentation Framework](#our-image-segmentation-framework)
2. [Code structure](#code-structure)
3. [Requirements](#requirements)
4. [Datasets](#datasets)
5. [Models](#models)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Reproducibility](#reproducibility)
9. [Pre-trained Models](#pre-trained-models)
10. [Results](#results)
11. [References](#references)

## Our Image Segmentation Framework
We conceived our framework to be able to handle both tensorflow-based models and torch-based models, as both are ubiquitous in the literature. To track our experiments, we used the MLFlow framework. Although it makes the code base a bit more complex, it allowed us to have a centralized space where we could log and **share** our experiments, model checkpoints, hyperparameters and so on. For hyperparameter tuning, we use [HyperOpt](http://hyperopt.github.io/hyperopt/), a framework conceived to optimize hyper parameter searches.

The overall structure of the repository is described below.

## Code structure

```
.
├── data_handling                   # Scripts for loading datasets
├── dataset                         # Folder containing the datasets
    ├── <dataset-name-1>
    ├── <dataset-name-2>  
    └── ...     
├── factory                         # Helper for Loading models
├── hyperopt_                       # Utilities for hyperparameter searches (using the HyperOpt framework)
├── losses                          # Folder with losses used across models
├── models                          # Folder with our various models
    ├── reinforcement               # RL models
    ├── Torch                       # Torch models
    └── TF                          # Tensorflow models
├── trainers                        # Custom trainers for each model
    ├── reinforcement               # Trainers for RL models
    ├── Torch                       # Trainers for Torch models
    └── TF                          # Trainers for Tensorflow models
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

When a model is trained, checkpoints appear in the `checkpoints` folder.

## Evaluation

#### Torch models
To evaluate a Torch model, run:

```eval
python torch_predictor.py -m <model-name> -c <checkpoint-name>
```

#### Tensorflow models
To evaluate a Tensorflow model, run:

```eval
python tf_predictor.py -m <model-name> -c <checkpoint-name>
```

#### RL models
To evaluate an RL model, run:

```eval
python todo
```

The predictions appear in the `output_preds` directory

## Reproducibility

To reproduce our results, run the following commands:

#### Baselines

1) U-Net:
```reproduce
python todo
```
2) U-Net3+:
```reproduce
python todo
```
3) CRA-Net:
```reproduce
python todo
```
4) DeepLabV3:
```reproduce
python todo
```
5) SegFormer:
```reproduce
python todo
```

#### Our Contributions

1) UNet Exp:
```reproduce
python todo
```
2) RL Seg:
```reproduce
python todo
```

#### Our Ensemble submission

Please follow the following steps:

1) Do blabla
```ensemble
python todo
```
2) Then do blabla
```ensemble
python todo
```
2) Finally do blabla
```ensemble
python todo
```

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

## References

This repository uses substantial code adapted from other repositories:

* [Attention UNet TF](https://github.com/ozan-oktay/Attention-Gated-Networks)
* [GL Dense UNet](https://github.com/cugxyy/GL-Dense-U-Net/blob/master/Model/GL_Dense_U_Net.py)
* [UNet TF](https://github.com/zhixuhao/unet)
* [UNet++ TF](https://github.com/MrGiovanni/UNetPlusPlus)
* [CRA Net](https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction)
* [DeepLabV3](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)
* [Fast SCNN](https://github.com/Tramac/Fast-SCNN-pytorch/blob/master/models/fast_scnn.py)
* [SegFormer](https://github.com/NVlabs/SegFormer/)
* [UNet Torch](https://github.com/milesial/Pytorch-UNet)
* [Lawin](https://github.com/yan-hao-tian/lawin/blob/main/lawin_head.py)
* [U2Net](https://github.com/xuebinqin/U-2-Net/tree/master/model)
