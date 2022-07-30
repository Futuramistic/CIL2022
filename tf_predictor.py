"""
Given a trained model, make predictions on the test set
"""
import torch
import pickle
import tensorflow.keras as K
import numpy as np
import argparse
import tensorflow as tf
import tensorflow_addons as tfa

from tqdm import tqdm
from factory import Factory
from losses.precision_recall_f1 import precision_recall_f1_score_tf
from utils import *
from losses.loss_harmonizer import DEFAULT_TF_DIM_LAYOUT, collapse_channel_dim_tf


model = None

# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
test_set_size = 144


def get_saliency_map(model, image):
    """
    Given a model and an image, compute its saliency map
    WARNING: Extremly high memory usage; unused in code due to this issue!
    !!! Should only be run on high-memory GPUs !!!
    Args:
        model: TF model
        image (Tensor): Image to compute the saliency map for
    Returns: 
        Tensor - saliency map of an image
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        
        loss = predictions
    
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    
    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normalize between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + K.backend.epsilon())
    
    saliency_image_gray =tf.expand_dims(tf.squeeze(tf.convert_to_tensor(smap)),-1)
    return tf.image.grayscale_to_rgb(saliency_image_gray)


def compute_best_threshold(loader, apply_sigmoid, with_augmentation=True, checkpoint=None):
    """
    Line search segmentation thresholds and select the one that works
    the best on the training set.
    Args:
        loader: the dataset loader
        apply_sigmoid (bool): whether to apply the sigmoid on the output of the model
        with_augmentation (bool): If true, augment the images, predict on each augmented version,
        then ensemble the predictions
        checkpoint (str): path to checkpoint of model (used for caching the best threshold)
    """
    cache_path = '.'.join(checkpoint.split('.')[:-1]) + '_best_threshold_cache.txt' if checkpoint is not None else None
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, 'r') as f:
            return float(f.read())
    
    best_thresh = 0
    best_f1_score = 0
    for thresh in np.linspace(0, 1, 41):
        f1_scores = []
        with torch.no_grad():
            for x_, y, *_ in tqdm(loader):
                if with_augmentation:
                    x_ = _augment(tf.squeeze(x_))

                # will not work on larger samples otherwise
                output_ = tf.concat([model.predict(tf.expand_dims(sample, 0), verbose=0) for sample in x_], axis=0)
                if type(output_) is tuple:
                    output_ = output_[0]
                
                # models not requiring "apply_sigmoid" will have applied it here already, so this should be
                # before the unification to ensure consistency
                if apply_sigmoid:
                    output_ = K.layers.Activation('sigmoid')(output_)

                output_ = tf.expand_dims(collapse_channel_dim_tf(output_, take_argmax=False),
                                         DEFAULT_TF_DIM_LAYOUT.index('C'))

                if with_augmentation:
                    output_ = _unify(tf.squeeze(output_))
                
                preds = output_ >= thresh
                _, _, _, _, _, _, _, f1_weighted, _, _, f1_patchified_weighted =\
                    precision_recall_f1_score_tf(tf.squeeze(preds), tf.squeeze(y))
                # change between f1_weighted and f1_patchified_weighted as appropriate
                f1_scores.append(f1_weighted)
                # preds = (output_ >= thresh).astype(np.int8)
                # *_, f1_score = precision_recall_f1_score_tf(preds, y)
                # f1_scores.append(f1_score.cpu().numpy())
                del x_
                del y
        f1_score = np.mean(f1_scores)
        print('Threshold', thresh, '- F1 score:', f1_score)
        if f1_score > best_f1_score:
            best_thresh = thresh
            best_f1_score = f1_score
    print('Best F1-score on train set:', best_f1_score, 'achieved with a threshold of:', best_thresh)

    if cache_path is not None:
        with open(cache_path, 'w') as f:
            f.write(str(best_thresh))

    return best_thresh


def h_flip_transform(image):
    """
    Flip the given image horizontally
    """
    image = tf.image.flip_left_right(image)
    return image


def _ensemble(images):
    """
    Average the input images along the first dimension
    Args:
        images (Tensor[N, H, W, C])
    Returns:
        an ensembled image (Tensor[1, H, W, C])
    """
    ensembled = tf.reduce_mean(images, axis=0, keepdims=True)
    return ensembled


def _augment(image):
    """
    Augment the input with the 90° rotated versions and the horizontal and vertical flipped versions
    Args:
        image (Tensor [1, H, W, C])
    Returns:
        augmented tensor (Tensor [8, H, W, C])
    """
    xs = []
    for i in range(4):
        xs.append(tf.image.rot90(image, i))
    flipped = tf.image.flip_up_down(image)
    for i in range(4):
        xs.append(tf.image.rot90(flipped, i))
    augmented = tf.stack(xs, axis=0)
    return augmented


def _unify(images):
    """
    Perform the inverse operation from augment. First the inverse transforms are applied,
    then the predictions are ensembled into one image.
    Args:
        images (Tensor [8, ..., H, W])
    Returns:
        a single ensembled image (Tensor [1, ..., H, W])
    """
    ll = []
    for i in range(4):
        img_i = tf.image.rot90(tf.expand_dims(images[i], -1), -i)
        ll.append(tf.squeeze(img_i))
    for i in range(4):
        img_i = tf.image.rot90(tf.expand_dims(images[i+4], -1), -i)
        ll.append(tf.squeeze(tf.image.flip_up_down(img_i)))
    images_new = tf.stack(ll, axis=0)
    return _ensemble(images_new)


def main():
    with_augmentation = True  # TODO: turn into command line arg

    parser = argparse.ArgumentParser(description='Predictions maker for tensorflow models')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('--apply_sigmoid', dest='apply_sigmoid', action='store_true', required=False)
    parser.add_argument('--saliency', dest='compute_saliency', action='store_true', required=False)
    parser.add_argument('-o', '--floating_output_dir', type=str,
                        help='The directory to store the output of the network to, if --floating_prediction is passed',
                        required=False, default=OUTPUT_FLOAT_DIR)
    parser.add_argument('--use_floating_output_cache', action='store_true', help='If specified, do not recompute floating predictions if they are found in floating_output_dir', required=False)
    parser.set_defaults(apply_sigmoid=False)
    parser.set_defaults(compute_saliency=False)
    parser.set_defaults(floating_prediction=False)
    parser.add_argument('--floating_prediction', dest='floating_prediction', action='store_true',
                        help='If specified, dump the output of the network to a pickle file, else the default '
                             'behavior is to threshold the output to get a binary prediction')
    options = parser.parse_args()

    model_name = options.model
    trained_model_path = options.checkpoint
    apply_sigmoid = options.apply_sigmoid
    compute_saliency = options.compute_saliency
    floating_output_dir = options.floating_output_dir
    floating_prediction = options.floating_prediction
    use_floating_output_cache = options.use_floating_output_cache

    os.makedirs(floating_output_dir, exist_ok=True)

    global model
    # Create loader, trainer etc. from factory
    factory = Factory.get_factory(model_name)
    dataloader = factory.get_dataloader_class()(dataset=dataset)
    model = factory.get_model_class()(input_shape=[400, 400, 3])
    trainer = factory.get_trainer_class()(dataloader=dataloader, model=model)

    # Load the trained model weights
    model.load_weights(trained_model_path)
    # Freeze layers
    for layer in model.layers: 
        layer.trainable = False
    preprocessing = trainer.preprocessing
    original_dataloader = factory.get_dataloader_class()(dataset='original')
    train_loader = original_dataloader.get_training_dataloader(split=0.174, batch_size=1, preprocessing=preprocessing)
    test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

    create_or_clean_directory(OUTPUT_PRED_DIR)

    train_bs = 16
    train_dataset_size, _, _ = dataloader.get_dataset_sizes(split=0.2)
    segmentation_threshold = compute_best_threshold(train_loader.take(train_dataset_size),
                                                    apply_sigmoid=apply_sigmoid,
                                                    checkpoint=trained_model_path)

    # Prediction
    i = 0
    for x in tqdm(test_loader):
        if i > test_set_size - 1:
            break
        
        if floating_prediction and use_floating_output_cache and os.path.isfile(f'{floating_output_dir}/satimage_{offset+i}.pkl'):
            i += 1
            continue
        
        if compute_saliency:
            K.preprocessing.image.save_img(f'{SALIENCY_MAP_DIR}/saliency_map_{offset+i}.png', get_saliency_map(model,x))

        if with_augmentation:
            x = _augment(tf.squeeze(x))
        
        output = model.predict(x)

        if apply_sigmoid:
            output = K.layers.Activation('sigmoid')(output)
        
        channel_dim_idx = DEFAULT_TF_DIM_LAYOUT.find('C')
        data_format = "channels_last" if channel_dim_idx == 3 else "channels_first"

        if output.shape[channel_dim_idx] > 1:
            output = collapse_channel_dim_tf(output, take_argmax=False)
            # Prior incorrect version (caused loss of precision for softmax-based models
            # due to premature rounding): 
            # output = np.argmax(output, axis=channel_dim_idx)
            # output = np.argmax(output, axis=channel_dim_idx)
            # output = np.expand_dims(output, axis=channel_dim_idx)

        if with_augmentation:
            output = tf.expand_dims(_unify(tf.squeeze(output)), -1)  # add channel dimension back in
        
        while len(output.shape) > 3:
            output = output[0]

        if floating_prediction:
            with open(f'{floating_output_dir}/satimage_{offset + i}.pkl', 'wb') as handle:
                array = np.squeeze(output.numpy())
                # rescale so that old optimal threshold is at 0.5
                array = (array < segmentation_threshold) * array / segmentation_threshold * 0.5 + \
                        (array >= segmentation_threshold) * \
                        ((array - segmentation_threshold) / (1 - segmentation_threshold) + 0.5)
                pickle.dump(array, handle)
        else:
            pred = tf.cast(output >= segmentation_threshold, tf.int32) * 255
            K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred, data_format=data_format)
        del x
        if i >= test_set_size - 1:
            break
        i += 1


if __name__ == '__main__':
    # # use CPU for prediction even though GPU is available (e.g. for memory reasons):
    # with tf.device('/cpu:0'):
    #    main()

    # # use GPU if available, else use CPU:
    main()
