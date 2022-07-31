# CIL Course Project: Road Segmentation

This repository contains the implementation of our solution for the Road Segmentation task of the Computational Intelligence Lab.

Please refer to the accompanying [temp report](./report.pdf) for an in-depth description of our 
research process and results.


<p align="center"><img src="http://kamui-interactive.com/wp-content/uploads/2022/07/unetexp.png" alt="Our UNet Exp architecture" width="600"></p>
<p align="center">Our UNet Exp architecture</p>

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
7. [Predicting](#predicting)
8. [The RL Framework](#the-rl-framework)
9. [Reproducibility](#reproducibility)
10. [Pre-trained Models](#pre-trained-models)
11. [Results](#results)
12. [References](#references)

## Our Image Segmentation Framework
We conceived our framework to be able to handle both tensorflow-based models and torch-based models, as both are ubiquitous in the literature. To track our experiments, we used the MLFlow framework. Although it makes the code base a bit more complex, it allowed us to have a centralized space where we could log and **share** our experiments, model checkpoints, hyperparameters and so on. For hyperparameter tuning, we use [HyperOpt](http://hyperopt.github.io/hyperopt/), a framework conceived to optimize hyper parameter searches.

The overall structure of the repository is described below.

## Code structure

```
.
├── adaboost                        # Module for running adaboost on models
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

- Linux:

```setup
pip install -r requirements_unix.txt
```

- Windows:
```setup
pip install -r requirements_windows.txt
```

- On Euler, load the following modules: 
```
module load gcc/8.2.0 && module load python_gpu/3.9.9 && module load openmpi && module load eth_proxy
```

## Datasets

#### Available datasets
For reproducing our results, you do not need to setup a dataset folder, you 
just need to specify the dataset name in the command line when training and 
our framework will automatically download the relevant dataset. Available 
dataset names are:

> * original: dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
> * original_split_1: validation split 1 of original dataset (samples "satimage_0.png" to "satimage_24.png" from "original" dataset
used as validation set)
> * original_split_2: validation split 2 of original dataset (samples "satimage_25.png" to "satimage_49.png" from "original" dataset
used as validation set)
> * original_split_3: validation split 3 of original dataset (samples "satimage_50.png" to "satimage_74.png" from "original" dataset
used as validation set)
> * original_split_2_aug_6: validation split 2 of original dataset (samples "satimage_25.png" to "satimage_49.png" from "original" dataset used as validation set), with augmented training set using Preprocessor (x6)
> * original_split_3_aug_6: validation split 3 of original dataset (samples "satimage_50.png" to "satimage_74.png" from "original" dataset used as validation set), with augmented training set using Preprocessor (x6)
> * maps_filtered: hand-filtered dataset of 1597 satellite images screenshotted from Google Maps
same 25 validation samples as in "new_original"
> * maps_filtered_no_original_aug_6:  maps_filtered, without any original samples from the training set,
400x400 but with augmented training set using Preprocessor (x6), same 25 validation samples as in "new_original"
> * maps_filtered_aug_6:  maps_filtered, with 119 samples from the training set,
400x400 but with augmented training set using Preprocessor (x6), same 25 validation samples as in "new_original"

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
- UNet (name: 'unettf')
- UNet++ (name: 'unet++')
- UNetExp (name: 'unetexp')
- UNet3+ (name: 'unet3+')
- Attention UNet++ ('attunet++')
- GL Dense UNet ('gldenseunet')
```

*Torch models*
```
- UNet (name: 'unet')
- CRA-Net (name: 'cranet')
- DeepLabV3 (name: 'deeplabv3')
- DeepLabV3+GAN (name: 'deeplabv3plusgan')
- SegFormer (name: 'segformer')
- TwoShotNet (name: 'twoshotnet')
- Lawin (name: 'lawin')
```

*Reinforcement Learning models (Torch)*
```
- SimpleRLCNN (name: 'simplerlcnn')
- SimpleRLCNNMinimal (name: 'simplerlcnnminimal')
```

## Training

To train the available models, run this command:

```train
python main.py -d <dataset-name> -m <model-name> --batch_size <batch-size> --num_epochs <num-epochs> -E <experiment-name>
```

For example:

```train
python main.py -d original -m deeplabv3 -E training_run --batch_size 4 --num_epochs 30
```

>📋  TODO put commands for our final models/RL models

When a model is trained, checkpoints appear in the `checkpoints` folder.

## Predicting

#### Torch models
To output predictions using a Torch model, run:

```eval
python torch_predictor.py -m <model-name> -c <checkpoint-name>
```
You can add extra arguments:
* `--floating_predictions`: to output non-thresholded value. If this is specified, also specify `--floating_output_dir <path/to/output/dir`
* `--apply_sigmoid`: this must specified for the following models: UNet, CRANet, DeepLabV3, SegFormer
* `--blob_threshold <integer>`: if specified, all 'blobs' with less than `<integer>` pixels will be removed from the segmentation

#### Tensorflow models
To output predictions using a Tensorflow model, run:

```eval
python tf_predictor.py -m <model-name> -c <checkpoint-name>
```
You can add extra arguments:
* `--floating_predictions`: to output non-thresholded value. If this is specified, also specify `--floating_output_dir <path/to/output/dir`
* `--blob_threshold <integer>`: if specified, all 'blobs' with less than `<integer>` pixels will be removed from the segmentation
* `--saliency`: if specified, saliency maps are outputted to the `saliency_maps` directory

#### RL models
To output predictions using an RL model, run:

```eval
python todo
```

The predictions appear in the `output_preds` directory

## The RL Framework
![The walker algorithm](http://kamui-interactive.com/wp-content/uploads/2022/07/walker_test_5.gif)

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
6) Lawin:

To obtain thes results, first pre-train model:
```
python main.py --model=lawin --dataset=maps_filtered --split=0.9846 -E=Lawin --num_epochs=5000 --checkpoint_interval=500 --hyper_seg_threshold=False '--run_name=Pretrain on maps_filtered' --use_geometric_augmentation=True --use_color_augmentation=True --blobs_removal_threshold=0 --use_channelwise_norm=True --backbone_name=mit_bfive
```
Then, to finetune on one of the split datasets, run this command with checkpoint provided in the --load_checkpoint_path argument:
```
python main.py --model=lawin --dataset=original_split_1 --split=0.827 -E=Lawin --num_epochs=5000 --checkpoint_interval=100000 --hyper_seg_threshold=False '--run_name=Finetune on original_split_1 (pretrained on maps_filtered)' --use_geometric_augmentation=True --use_color_augmentation=True --blobs_removal_threshold=0 --use_channelwise_norm=True --backbone_name=mit_bfive --load_checkpoint_path=<MODEL/URL/OR/PATH/HERE>
```

#### Our Contributions

1) UNet Exp:
To obtain thes results, first pre-train model:
```
python main.py --model=unetexp --dataset=maps_filtered --split=0.9846 -E=UnetExp --num_epochs=200 --hyper_seg_threshold=False '--run_name=Pretrain on maps_filtered' --use_geometric_augmentation=True --batch_size=4 
```
Then, as with previous models, run this command with checkpoint provided in the --load_checkpoint_path argument on the selected finetuning dataset:
```
python main.py --model=unetexp --dataset=original_split_1 --split=0.827 -E=UnetExp --num_epochs=200 --hyper_seg_threshold=False '--run_name=Finetune on original_split_1 (pretrained on maps_filtered)' --use_geometric_augmentation=True --batch_size=4 --load_checkpoint_path=<MODEL/URL/OR/PATH/HERE>
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

You can download pretrained models here:

- Models used for the final submission:
  - [Lawin 1](https://polybox.ethz.ch/index.php/s/VI7G1l0B2SkKI3N/download) trained on "maps_filtered" and fine-tuned on "original_split_1", "original_split_2" and "original_split_3"
  - [Lawin 2](https://polybox.ethz.ch/index.php/s/UDgZiDWdUt8Yof0/download) trained on "maps_filtered" and fine-tuned on "original_split_1", "original_split_2" and "original_split_3"
  - [Lawin 3](/download) trained on "maps_filtered" and fine-tuned on "original_split_1", "original_split_2" and "original_split_3"

## Results

Our models achieve the following performance on the 'original' dataset:

| Model name           | Road F1-Score     | Macro F1-Score     | Weighted F1-Score |
| -------------------- | ----------------- | ------------------ | ----------------- |
| UNet Exp             |     0.999         |          1         |         1         |
| Submission ensemble  |     0.998         |          1         |         1         |

Our ensemble of Lawin models extensively pretrained on the scraped dataset ("maps_filtered") and fine-tuned on the original dataset ("original_split_1", "original_split_2", "original_split_3") achieved an F1-Score of `0.93737` on the Kaggle competition, granting us the `3rd` position on the Leaderboard. 

## References

This repository uses substantial code adapted from other repositories:

* [Attention UNet TF](https://github.com/ozan-oktay/Attention-Gated-Networks)
* [GL Dense UNet](https://github.com/cugxyy/GL-Dense-U-Net/blob/master/Model/GL_Dense_U_Net.py)
* [UNet TF](https://github.com/zhixuhao/unet)
* [UNet++ TF](https://github.com/MrGiovanni/UNetPlusPlus)
* [CRA Net](https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction)
* [DeepLabV3](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)
* [SegFormer](https://github.com/NVlabs/SegFormer/)
* [UNet Torch](https://github.com/milesial/Pytorch-UNet)
* [Lawin](https://github.com/yan-hao-tian/lawin/blob/main/lawin_head.py)
* [U2Net](https://github.com/xuebinqin/U-2-Net/tree/master/model)
