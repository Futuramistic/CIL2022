# Imports
import torch
from tqdm import tqdm

from factory import Factory
from losses.precision_recall_f1 import precision_recall_f1_score_tf
from models.TF.UNetTF import UNetTF
from data_handling.dataloader_tf import TFDataLoader
from trainers.UnetTF import UNetTFTrainer
import tensorflow as tf
import tensorflow.keras as K
from utils import *
import numpy as np
from losses.loss_harmonizer import DEFAULT_TF_DIM_LAYOUT


compute_best_threshold_split = 0.99  # ensure validation dataset has at least 1 sample

# modify in tf_predictor.py as well!
def compute_best_threshold(loader, apply_sigmoid):
    best_thresh = 0
    best_f1_score = 0
    for thresh in np.linspace(0, 1, 21):
        f1_scores = []
        with torch.no_grad():
            for x_, y in loader:
                output_ = model.predict(x_, verbose=0)
                if type(output_) is tuple:
                    output_ = output_[0]
                if apply_sigmoid:
                    output_ = K.layers.Activation('sigmoid')(output_)
                preds = (output_ >= thresh).astype(np.int8)
                _, _, f1_score = precision_recall_f1_score_tf(preds, y)
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


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
test_set_size = 144

# Parameters
model_name = 'gldenseunet'                                           # <<<<<<<<<<<<<<<<<< Insert model type
trained_model_path = 'original_checkpoint_0cddd7aeff8408184874efc38e60ae52.ckpt'       # <<<<<<<<<<<<<<<<<< Insert trained model name
apply_sigmoid = False                                                # <<<<<<<<<<<<<<<< Specify whether Sigmoid should
                                                                    # be applied to the model's output

# Create loader, trainer etc. from factory
factory = Factory.get_factory(model_name)
dataloader = factory.get_dataloader_class()(dataset=dataset)
model = factory.get_model_class()(input_shape=[400,400,3])
trainer = factory.get_trainer_class()(dataloader=dataloader, model=model)

# Load the trained model weights
model.load_weights(trained_model_path)

preprocessing = trainer.preprocessing
train_loader = dataloader.get_training_dataloader(split=0.99, batch_size=1, preprocessing=preprocessing)
test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

create_or_clean_directory(OUTPUT_PRED_DIR)

train_bs = 16
train_dataset_size, _, _ = dataloader.get_dataset_sizes(split=compute_best_threshold_split)
segmentation_threshold = compute_best_threshold(train_loader.take((train_dataset_size // train_bs) * train_bs),
                                                apply_sigmoid=apply_sigmoid)
# segmentation_threshold = 0.5

# Prediction
i = 0
for x in tqdm(test_loader):
    output = model.predict(x)
    channel_dim_idx = DEFAULT_TF_DIM_LAYOUT.find('C')
    data_format = "channels_last" if channel_dim_idx == 3 else "channels_first"
    if output.shape[channel_dim_idx] > 1:
        output = np.argmax(output, axis=channel_dim_idx)
        output = np.expand_dims(output, axis=channel_dim_idx)
    while len(output.shape) > 3:
        output = output[0]
    pred = (output >= segmentation_threshold).astype(int) * 255
    K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred, data_format=data_format)
    del x
    if i >= test_set_size - 1:
        break
    i += 1



