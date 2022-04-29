# Imports
import torch
from tqdm import tqdm
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

# Parameters
model = UNetTF()
trained_model_path = 'cp_ep-00000_it-00210_step-210.ckpt'  # Name of the pretrained model
segmentation_threshold = 0.5

# Load the model/data
model.load_weights(trained_model_path)

dataloader = TFDataLoader(dataset=dataset)
trainer = UNetTFTrainer(dataloader, model)
preprocessing = trainer.preprocessing

test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

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



