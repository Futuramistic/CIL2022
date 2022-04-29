# Imports
import torch
from tqdm import tqdm

from factory import Factory
from models.TF.UNetTF import UNetTF
from data_handling.dataloader_tf import TFDataLoader
from trainers.UnetTF import UNetTFTrainer
import tensorflow.keras as K
from utils import *
import numpy as np
from losses.loss_harmonizer import DEFAULT_TF_DIM_LAYOUT


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
test_set_size = 144
segmentation_threshold = 0.5

# Parameters
model_name = 'unettf'                                           # <<<<<<<<<<<<<<<<<< Insert model type
trained_model_path = 'cp_ep-00000_it-00210_step-210.ckpt'       # <<<<<<<<<<<<<<<<<< Insert trained model name

# Create loader, trainer etc. from factory
factory = Factory.get_factory(model_name)
dataloader = factory.get_dataloader_class()(dataset=dataset)
model = factory.get_model_class()()
trainer = factory.get_trainer_class()(dataloader=dataloader, model=model)

# Load the trained model weights
model.load_weights(trained_model_path)

preprocessing = trainer.preprocessing
test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

create_or_clean_directory(OUTPUT_PRED_DIR)

# Prediction
for i, x in tqdm(enumerate(test_loader)):
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



