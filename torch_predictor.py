# Imports
import torch
import numpy as np
import tensorflow.keras as K
import argparse
import torchvision.transforms.functional as TF

from factory import Factory
from losses.precision_recall_f1 import precision_recall_f1_score_torch
from blobs_remover import remove_blobs
from tqdm import tqdm
from losses import *
from utils import *


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'  # dataset name
sigmoid = torch.nn.Sigmoid()

device = None
model = None
test_loader = None
blob_threshold = None


"""
Given a trained model, make predictions on the test set
"""


def compute_best_threshold(loader, apply_sigmoid):
    """
    Line search segmentation thresholds and select the one that works
    the best on the training set.
    Args:
        loader: the dataset loader
        apply_sigmoid (bool): whether to apply the sigmoid on the output of the model
    """
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
    """
    Make predictions using a trained model
    Args:
        segmentation_threshold (float): the threshold determining the boundary between pixel classes
        apply_sigmoid (bool): whether to apply the sigmoid on the output of the model
        with_augmentation (bool): If true, augment the images, predict on each augmented version,
        then ensemble the predictions
    """
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
            while len(pred.shape) == 2:
                pred = pred[None, :, :]
            K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred,
                                           data_format="channels_first")
            i += 1
            del x


# Transformation functions
def rotation_transform(image, angle, inverse=False):
    """
    Apply a rotation transformation on an image
    Args:
        image (Tensor): Input image
        angle (float): rotation angle
        inverse (bool): If true apply the inverse transformation
    """
    if not inverse:
        image = TF.rotate(image, angle)
    else:
        image = TF.rotate(image, -angle)
    return image


def v_flip_transform(image):
    """
    Flip the given image vertically
    """
    image = TF.vflip(image)
    return image


def h_flip_transform(image):
    """
    Flip the given image horizontally
    """
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


def main():
    parser = argparse.ArgumentParser(description='Predictions maker for torch models')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('--apply_sigmoid', dest='apply_sigmoid', action='store_true', required=False)
    parser.add_argument('--blob_threshold', type=int, default=250, help='The threshold for the blob processing')
    parser.set_defaults(apply_sigmoid=False)
    options = parser.parse_args()

    global blob_threshold, test_loader, model, device

    model_name = options.model
    trained_model_path = options.checkpoint
    apply_sigmoid = options.apply_sigmoid
    blob_threshold = options.blob_threshold

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

    original_dataloader = factory.get_dataloader_class()(dataset='original')
    train_loader = original_dataloader.get_training_dataloader(split=0.174, batch_size=1, preprocessing=preprocessing)
    test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

    create_or_clean_directory(OUTPUT_PRED_DIR)

    # Compute the best threshold
    segmentation_threshold = compute_best_threshold(train_loader, apply_sigmoid=apply_sigmoid)

    # Make the final predictions
    predict(segmentation_threshold, apply_sigmoid=apply_sigmoid, with_augmentation=False)


if __name__ == '__main__':
    main()
