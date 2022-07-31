# CIL Course Project: Road Segmentation

This repository contains the implementation of our solution for the Road Segmentation task of the Computational Intelligence Lab.

Please refer to the accompanying [temp report](./report.pdf) for an in-depth description of our 
research process and results.


<p align="center"><img src="http://kamui-interactive.com/wp-content/uploads/2022/07/unetexp.png" alt="Our U-Net Exp architecture" width="600"></p>
<p align="center">Our U-Net Exp architecture</p>

## Authors
Sorted alphabetically:
* Alexey Gavryushin
* Noureddine Gueddach
* Anne Marx
* Mateusz Nowak

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
We conceived our framework to be able to handle both TensorFlow-based models and Torch-based models, as both are ubiquitous in the literature. To track our experiments, we used the [MLflow](https://mlflow.org/) framework. Although it makes the code base a bit more complex, it allowed us to have a centralized space where we could log and **share** our experiments, model checkpoints, hyperparameters, etc. For hyperparameter tuning, we use [HyperOpt](http://hyperopt.github.io/hyperopt/), a framework conceived to optimize hyperparameter searches.

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
    ├── Torch                       # PyTorch models
    └── TF                          # Tensorflow models
├── trainers                        # Custom trainers for each model
    ├── reinforcement               # Trainers for RL models
    ├── Torch                       # Trainers for PyTorch models
    └── TF                          # Trainers for Tensorflow models
├── utils                           # Utility functions and constants
└── processing                      # Contains processing files such as preprocessing, downloading etc.
torch_predictor.py                  # Script for making predictions on a PyTorch model
tf_predictor.py                     # Script for making predictions on a TensorFlow model
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

- On [Euler](https://scicomp.ethz.ch/wiki/Euler), load the following modules: 
```
module load gcc/8.2.0 && module load python_gpu/3.9.9 && module load openmpi && module load eth_proxy
```

## Datasets

#### Available datasets
For reproducing our results, you do not need to setup a dataset folder, you 
just need to specify the dataset name in the command line when training and 
our framework will automatically download the relevant dataset. Available 
datasets are:

> * `original`: dataset used in the ETHZ CIL Road Segmentation 2022 Kaggle competition
> * `original_split_1`: validation split 1 of original dataset (samples `satimage_0.png` to `satimage_24.png` from `original` dataset
used as validation set)
> * `original_split_2`: validation split 2 of original dataset (samples `satimage_25.png` to `satimage_49.png` from `original` dataset
used as validation set)
> * `original_split_3`: validation split 3 of original dataset (samples `satimage_50.png` to `satimage_74.png` from `original` dataset
used as validation set)
> * `original_split_2_aug_6`: validation split 2 of original dataset (samples `satimage_25.png` to `satimage_49.png` from `original` dataset used as validation set), with augmented training set using `preprocessor.py` (x6)
> * `original_split_3_aug_6`: validation split 3 of original dataset (samples `satimage_50.png` to `satimage_74.png` from `original` dataset used as validation set), with augmented training set using `preprocessor.py` (x6)
> * `maps_filtered`: hand-filtered dataset of 1597 satellite images screenshotted from Google Maps, plus samples "satimage_25.png" to "satimage_143.png" from "original" dataset
same 25 validation samples as in "new_original"
> * `maps_filtered_no_original_aug_6`:  maps_filtered, without any original samples from the training set,
400x400 but with augmented training set using `preprocessor.py` (x6), same 25 validation samples as in `new_original`
> * `maps_filtered_aug_6`:  maps_filtered, with 119 samples from the training set,
400x400 but with augmented training set using `preprocessor.py` (x6), same 25 validation samples as in `new_original`

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
We worked with multiple models, the code for all of which is available in the `models` folder.
The available models are (with the respective name to pass to the training
command-line below under the `model-name` argument):

*TensorFlow models*
```
- U-Net (name: 'unettf')
- UNet++ (name: 'unet++')
- U-Net Exp (name: 'unetexp')
- UNet 3+ (name: 'unet3+')
- Attention UNet++ ('attunet++')
- GL-Dense-U-Net ('gldenseunet')
```

*Torch models*
```
- U-Net (name: 'unet')
- CRA-Net (name: 'cranet')
- DeepLabV3 (name: 'deeplabv3')
- DeepLabV3+GAN (name: 'deeplabv3plusgan')
- TwoShotNet (name: 'twoshotnet')
- SegFormer (name: 'segformer')
- Lawin (name: 'lawin')
```

*Reinforcement Learning models (Torch)*
```
- SimpleRLCNN (name: 'simplerlcnn')
- SimpleRLCNNMinimal (name: 'simplerlcnnminimal')
- SimpleRLCNNMinimalSupervised (name: 'simplerlcnnminimalsupervised')
```

## Training

To train the available models, run this command:

```train
python main.py -d <dataset-name> -m <model-name> -E <experiment-name> --batch_size <batch-size> --num_epochs <num-epochs>
```

For example:

```train
python main.py -d original -m deeplabv3 -E training_run --batch_size 4 --num_epochs 30
```

When a model is trained, checkpoints appear in the `checkpoints` folder.

## Predicting

### PyTorch models
To output predictions using a PyTorch model, run:

```eval
python torch_predictor.py -m <model-name> -c <checkpoint-name>
```
You can add extra arguments:
* `--floating_predictions`: to output non-thresholded values. If this is specified, also specify `--floating_output_dir <path/to/output/dir>`
* `--apply_sigmoid`: this must specified for the following models: UNet, CRANet, DeepLabV3
* `--blob_threshold <integer>`: if specified, all 'blobs' with less than `<integer>` pixels will be removed from the segmentation

### TensorFlow models
To output predictions using a TensorFlow model, run:

```eval
python tf_predictor.py -m <model-name> -c <checkpoint-name>
```
You can add extra arguments:
* `--floating_predictions`: to output non-thresholded value. If this is specified, also specify `--floating_output_dir <path/to/output/dir`
* `--blob_threshold <integer>`: if specified, all 'blobs' with less than `<integer>` pixels will be removed from the segmentation
* `--saliency`: if specified, saliency maps are saved to the `saliency_maps` directory


## The RL Framework

#### Unsupervised setting

Example commands can be found in `docs/commands.md`. See docstrings in `models/reinforcement/first_try.py` and `models/reinforcement/environment.py` for details.

#### Supervised setting

![The Irresistable Hitchhiker Algorithm in action](http://kamui-interactive.com/wp-content/uploads/2022/07/walker_test_5.gif)

Please run `python -m processing.opt_brush_radius_calculator` on the desired dataset, e.g. `python -m processing.opt_brush_radius_calculator --dataset=new_original`.
Then run `python -m processing.non_maximum_suppressor` on the desired dataset, e.g. `python -m processing.non_maximum_suppressor --dataset=new_original`. 

Visualizations will be logged to MLflow at `http://algvrithm.com:8000/`. U/P are `cil22` and `equilibrium`, respectively.

Train the network using the commands in `docs/commands.md`.

## MonoBoost

Example command to run MonoBoost:

```
python3 main.py --model=deeplabv3 --use_adaboost=True --monoboost=True --adaboost_runs=100 --dataset=new_original --experiment_name=MonoBoost_Test --split=0.827 --num_epochs=35 --checkpoint_interval=700 --hyper_seg_threshold=True --optimizer_or_lr=0.0002 --use_geometric_augmentation=True --use_color_augmentation=True --monoboost_temperature=0.1
```

See `adaboost/adaboost.py` for details. Training details will be logged to MLflow at `http://algvrithm.com:8000/`. U/P are `cil22` and `equilibrium`, respectively.


## Reproducibility

To reproduce our results, run the following commands:

#### Baselines

1) U-Net3+:
```
python main.py --experiment_name=UNet3+ --model=unet3+ --architecture=vgg --kernel_regularizer=None --dropout=0.0 --normalize=True --optimizer_or_lr=1e-3 --dataset=original_split_1 --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=False --batch_size=2 --num_epochs=500 --checkpoint_interval=0 --hyper_seg_threshold=True --blobs_removal_threshold=0
```
2) DeepLabV3:
```
python main.py --experiment_name=DeepLabV3 --model=deeplabv3 --dataset=original_split_1 --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=True --batch_size=2 --optimizer_or_lr=2e-4 --num_epochs=200 --checkpoint_interval=0 --hyper_seg_threshold=True --blobs_removal_threshold=0
```
3) SegFormer:
```
python main.py --experiment_name=SegFormer --model=segformer --backbone_name=mit_bfive --dataset=original_split_1 --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=True --num_epochs=1000 --checkpoint_interval=0 --hyper_seg_threshold=True  --blobs_removal_threshold=0
```
4) Lawin:
```
python3 main.py --experiment_name=Lawin --model=lawin --backbone_name=mit_bfive --dataset=original_split_1 --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=True  --use_channelwise_norm=True --num_epochs=300 --checkpoint_interval=0 --hyper_seg_threshold=False --blobs_removal_threshold=0
```

To obtain our results, first pre-train the model (e.g. for Lawin):
```
python main.py --experiment_name=Lawin --model=lawin --backbone_name=mit_bfive --dataset=maps_filtered --split=0.9846 --use_geometric_augmentation=True --use_color_augmentation=True --use_channelwise_norm=True --num_epochs=5000 --checkpoint_interval=100000 --hyper_seg_threshold=False --blobs_removal_threshold=0
```
Then, to finetune on one of the split datasets, run this command with the path to the checkpoint provided in the `--load_checkpoint_path` argument (e.g. for Lawin):
```
python main.py --experiment_name=Lawin --model=lawin --backbone_name=mit_bfive --dataset=<original_split_1/original_split_2/original_split_3> --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=True --use_channelwise_norm=True --num_epochs=5000 --checkpoint_interval=100000 --hyper_seg_threshold=False --blobs_removal_threshold=0 --load_checkpoint_path=<MODEL/CHECKPOINT/PATH/HERE>
```

#### Our Contribution

U-Net Exp:
To obtain our results, first pre-train the model:
```
python main.py --experiment_name=UnetExp --model=unetexp --dataset=maps_filtered_aug_6 --split=0.9978 --num_epochs=200 --hyper_seg_threshold=False --use_geometric_augmentation=True --batch_size=4 
```
Then, as with previous models, run this command with checkpoint provided in the --load_checkpoint_path argument on the selected finetuning dataset:
```
python main.py --model=unetexp --dataset=original_split_1 --split=0.827 -E=UnetExp --num_epochs=200 --hyper_seg_threshold=False --use_geometric_augmentation=True --batch_size=4 --load_checkpoint_path=<MODEL/PATH/HERE>
```

#### Ensembling Submissions

We use ensembling to improve the performance of our submission. Please follow the following steps:

1) Train the models you wish to ensemble, with checkpointing enabled
2) Put the model names and their checkpoint paths into `sftp_paths.txt`, using the format described therein
3) Run `checkpoint_ensembler.py`
4) After completion, the resulting predictions should appear in the file `floating_ensemble.csv`

#### Our ensemble submission

We use an ensemble of Lawin transformers all pretrained on `maps_filtered` and then individually finetuned on `original_split_1`, `original_split_2` and `original_split_3`, respectively. Please follow the following steps to reproduce:

1) Pretrain a Lawin model on `maps_filtered`:
```
python3 main.py --experiment_name=Lawin --model=lawin --backbone_name=mit_bfive --dataset=maps_filtered --split=0.9846 --use_geometric_augmentation=True --use_color_augmentation=True  --use_channelwise_norm=True --num_epochs=300 --checkpoint_interval=0 --hyper_seg_threshold=False --blobs_removal_threshold=0
```
2) Finetune 3 models individually on the `original_split_1`, `original_split_2` and `original_split_3` datasets, respectively, e.g. for `original_split_1`:
```
python3 main.py --experiment_name=Lawin --model=lawin --backbone_name=mit_bfive --dataset=original_split_1 --split=0.827 --use_geometric_augmentation=True --use_color_augmentation=True  --use_channelwise_norm=True --num_epochs=300 --checkpoint_interval=0 --hyper_seg_threshold=False --blobs_removal_threshold=0 --load_checkpoint_path=<MODEL/PATH/HERE>
```
3) Put the path to each model's best checkpoint into `sftp_paths.txt`, then call `checkpoint_ensembler.py`

For easier reproduction, SFTP paths to the checkpoints we used to create our ensemble with `checkpoint_ensembler.py` are provided in `sftp_paths.txt`.

## Pre-Trained Models

You can download pretrained models here:

- Models used for ensembling the final submission using `checkpoint_ensembler.py`:
  - [Lawin 1](https://polybox.ethz.ch/index.php/s/UDgZiDWdUt8Yof0/download) pretrained on `maps_filtered` and finetuned on `original_split_1`
  - [Lawin 2](https://polybox.ethz.ch/index.php/s/VI7G1l0B2SkKI3N/download) pretrained on `maps_filtered` and finetuned on `original_split_2`
  - [Lawin 3](https://polybox.ethz.ch/index.php/s/Au4PRgKZynB16wG/download) pretrained on `maps_filtered` and finetuned on `original_split_3`
- U-NetExp Model:
  - [U-Net Exp](https://polybox.ethz.ch/index.php/s/lOsk3fX4HMI0NEP/download) trained on `maps_filtered_aug_6`

## Results

Our models achieve the following scores on the 'original_split_1' dataset:

| Model name            | Road F1 score     | Macro F1 score     | Weighted F1 score |
| --------------------- | ----------------- | ------------------ | ----------------- |
| U-Net Exp             |       0.727       |        0.838       |        0.920      |
| U-Net Exp *           |       0.755       |        0.856       |        0.931      |
| Lawin                 |       0.772       |        0.865       |        0.932      |
| Lawin 1 of ensemble * |       0.792       |        0.879       |        0.944      |


\* indicates models pretrained on `maps_filtered`. The backbones of all models in the table were initialized with weights obtained by pretraining on ImageNet prior to any training performed by us.

Our ensemble of Lawin models extensively pretrained on the scraped dataset (`maps_filtered`) and finetuned on the original dataset (`original_split_1`, `original_split_2`, `original_split_3`) achieved a weighted F1 score of `0.93737` on the public test set of the Kaggle competition, granting us the `4th` position on the leaderboard. We report only model 1's performance on `original_split_1` here, as it is the only model in the ensemble that never saw the validation samples of `original_split_1` during training.


## References

This repository relies on code adapted from other repositories:

* [Attention U-Net TF](https://github.com/ozan-oktay/Attention-Gated-Networks)
* [U-Net TF](https://github.com/zhixuhao/unet)
* [U-Net Torch](https://github.com/milesial/Pytorch-UNet)
* [U2Net](https://github.com/xuebinqin/U-2-Net/tree/master/model)
* [UNet++ TF](https://github.com/MrGiovanni/UNetPlusPlus)
* [GL-Dense-U-Net](https://github.com/cugxyy/GL-Dense-U-Net/blob/master/Model/GL_Dense_U_Net.py)
* [CRA-Net](https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction)
* [DeepLabV3](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)
* [SegFormer](https://github.com/NVlabs/SegFormer/)
* [Lawin](https://github.com/yan-hao-tian/lawin/blob/main/lawin_head.py)
