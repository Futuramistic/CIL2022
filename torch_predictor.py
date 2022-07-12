# Imports
import torch
from tqdm import tqdm
import pexpect

import torchvision.transforms.functional as TF
from factory import Factory
from losses.precision_recall_f1 import precision_recall_f1_score_torch
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from data_handling.dataloader_torch import TorchDataLoader
from trainers.u_net import UNetTrainer
from utils import *
from blobs_remover import remove_blobs

from losses import *
import numpy as np
import tensorflow.keras as K
from utils import *


# modify in tf_predictor.py as well!
def compute_best_threshold(loader, apply_sigmoid):
    best_thresh = 0
    best_f1_score = 0
    for thresh in np.linspace(0, 1, 21):
        f1_scores = []
        with torch.no_grad():
            for (x_, y) in tqdm(loader):
                x_ = x_.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                output_ = model(x_)
                if type(output_) is tuple:
                    output_ = output_[0]
                if apply_sigmoid:
                    output_ = sigmoid(output_)
                preds = (output_ >= thresh).float()
                _, _, f1_score = precision_recall_f1_score_torch(preds, y)
                f1_scores.append(f1_score.cpu().numpy())
                del x_
                del y
        f1_score = np.mean(f1_scores)
        print('Threshold', thresh, '- F1 score:', f1_score)
        if f1_score > best_f1_score:
            best_thresh = thresh
            best_f1_score = f1_score
    print('Best F1-score on train set:', best_f1_score, 'achieved with a threshold of:', best_thresh)
    return best_thresh


def predict(segmentation_threshold, apply_sigmoid, with_augmentation=False):
    # Prediction
    with torch.no_grad():
        i = 0
        for x in tqdm(test_loader):
            x = x.to(device, dtype=torch.float32)
            if with_augmentation:
                x = _augment(x)
            output = model(x)
            if with_augmentation:
                output = _unify(output)
            if type(output) is tuple:
                output = output[0]
            if apply_sigmoid:
                output = sigmoid(output)
            pred = (output >= segmentation_threshold).cpu().detach().numpy().astype(int)
            while len(pred.shape) > 3:
                pred = pred[0]
            pred = remove_blobs(pred, threshold=blob_threshold)
            pred *= 255
            K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred, data_format="channels_first")
            i += 1
            del x


# Transformation functions
def rotation_transform(image, angle, inverse=False):
    if not inverse:
        image = TF.rotate(image, angle)
    else:
        image = TF.rotate(image, -angle)
    return image


def v_flip_transform(image):
    image = TF.vflip(image)
    return image


def h_flip_transform(image):
    image = TF.hflip(image)
    return image


def _ensemble(images):
    """
    Average the input images along the first dimension
    Args:
        images (Tensor[..., ..., H, W])
    Returns:
        an ensembled image (Tensor[1, ..., H, W])
    """
    ensembled = torch.mean(images, dim=0, keepdim=True)
    return ensembled


def _augment(image):
    """
    Augment the input with the 90° rotated versions and the horizontal and vertical flipped versions
    Args:
        image (Tensor [1, ..., H, W])
    Returns:
        augmented tensor (Tensor [6, ..., H, W])
    """
    xs = []
    for i in range(4):
        xs.append(rotation_transform(image, i * 90, inverse=False))
    flipped = v_flip_transform(image)
    for i in range(4):
        xs.append(rotation_transform(flipped, i * 90, inverse=False))
    augmented = torch.cat(xs, dim=0)
    return augmented


def _unify(images):
    """
    Perform the inverse operation from augment. First the inverse transforms are applied,
    then the predictions are ensembled into one image.
    Args:
        images (Tensor [6, ..., H, W])
    Returns:
        a single ensembled image (Tensor [1, ..., H, W])
    """
    ll = []
    for i in range(4):
        img_i = torch.unsqueeze(rotation_transform(images[i], i * 90, inverse=True), dim=0)
        ll.append(img_i)
    for i in range(4):
        img_i = rotation_transform(images[i+4], i * 90, inverse=True)
        ll.append(torch.unsqueeze(v_flip_transform(img_i), dim=0))
    images = torch.cat(ll, dim=0)
    return _ensemble(images)


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
sigmoid = torch.nn.Sigmoid()

# Parameters
blob_threshold = 250
model_name = 'deeplabv3'                               # <<<<<<<<<<<<<<<<<< Insert model type
trained_model_path = 'cp_final_dlv.pt'                      # <<<<<<<<<<<<<<<<<< Insert trained model name
apply_sigmoid = True                                   # <<<<<<<<<<<<<<<< Specify whether Sigmoid should
                                                            # be applied to the model's output

# Create loader, trainer etc. from factory
factory = Factory.get_factory(model_name)
dataloader = factory.get_dataloader_class()(dataset=dataset)
model = factory.get_model_class()()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
trainer_class = factory.get_trainer_class()
trainer = trainer_class(dataloader=dataloader, model=model)  # , load_checkpoint_path=trained_model_path)

# Load the trained model weights
if torch.cuda.is_available():
    model_data = torch.load(trained_model_path)
else:
    model_data = torch.load(trained_model_path, map_location=device)
model.load_state_dict(model_data['model'])
model.eval()

preprocessing = trainer.preprocessing
train_loader = dataloader.get_training_dataloader(split=1, batch_size=1, preprocessing=preprocessing)
test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

create_or_clean_directory(OUTPUT_PRED_DIR)

segmentation_threshold = 0.45  # compute_best_threshold(train_loader, apply_sigmoid=apply_sigmoid)

predict(segmentation_threshold, apply_sigmoid=apply_sigmoid, with_augmentation=False)

