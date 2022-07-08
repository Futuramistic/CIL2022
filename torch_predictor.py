# Imports
import torch
from tqdm import tqdm
import pexpect

from factory import Factory
from losses.precision_recall_f1 import precision_recall_f1_score_torch
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from data_handling.dataloader_torch import TorchDataLoader
from trainers.u_net import UNetTrainer
import tensorflow.keras as K
from utils import *

import abc
import inspect
from losses import *
import mlflow
import numpy as np
import os
import pexpect
import paramiko
import pysftp
import requests
import shutil
import socket
import tensorflow.keras as K
import time

from data_handling import DataLoader
from requests.auth import HTTPBasicAuth
from utils import *
from utils.logging import mlflow_logger, optim_hyparam_serializer


# mlflow_experiment_name = 'retrieval'
# mlflow_experiment_id = None


# Doesn't work
# def init_mlflow():
#     is_windows = os.name == 'nt'
#     def add_known_hosts(host, user, password, jump_host=None):
#         spawn_str = \
#             'ssh %s@%s' % (user, host) if jump_host is None else 'ssh -J %s %s@%s' % (jump_host, user, host)
#         if is_windows:
#             # pexpect.spawn not supported on windows
#             import wexpect
#             child = wexpect.spawn(spawn_str)
#         else:
#             child = pexpect.spawn(spawn_str)
#         i = child.expect(['.*ssword.*', '.*(yes/no).*'])
#         if i == 1:
#             child.sendline('yes')
#             child.expect('.*ssword.*')
#         child.sendline(password)
#         child.expect('.*')
#         time.sleep(1)
#         child.sendline('exit')
#
#     mlflow_init_successful = True
#     MLFLOW_INIT_ERROR_MSG = 'MLflow initialization failed. Will not use MLflow for this run.'
#
#     try:
#         os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_HTTP_USER
#         os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_HTTP_PASS
#
#         mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL,
#                                        auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'],
#                                                           os.environ['MLFLOW_TRACKING_PASSWORD'])).text
#         try:
#             add_known_hosts(MLFLOW_HOST, MLFLOW_FTP_USER, mlflow_ftp_pass)
#         except:
#             add_known_hosts(MLFLOW_HOST, MLFLOW_FTP_USER, mlflow_ftp_pass, MLFLOW_JUMP_HOST)
#     except:
#         mlflow_init_successful = False
#         print(MLFLOW_INIT_ERROR_MSG)
#
#     if mlflow_init_successful:
#         try:
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#             experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
#             global mlflow_experiment_id
#             if experiment is None:
#                 mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
#             else:
#                 mlflow_experiment_id = experiment.experiment_id
#         except:
#             mlflow_init_successful = False
#             print(MLFLOW_INIT_ERROR_MSG)
#
#     return mlflow_init_successful
#
#
# if not init_mlflow():
#     print('mlflow initialization failed')


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


def predict(segmentation_threshold, apply_sigmoid):
    # Prediction
    with torch.no_grad():
        i = 0
        for x in tqdm(test_loader):
            x = x.to(device, dtype=torch.float32)
            output = model(x)
            if type(output) is tuple:
                output = output[0]
            if apply_sigmoid:
                output = sigmoid(output)
            pred = (output >= segmentation_threshold).cpu().detach().numpy().astype(int) * 255
            while len(pred.shape) > 3:
                pred = pred[0]
            K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred, data_format="channels_first")
            i += 1
            del x


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
sigmoid = torch.nn.Sigmoid()

# Parameters
model_name = 'unet'                                             # <<<<<<<<<<<<<<<<<< Insert model type
trained_model_path = 'cp_final.pt'                                  # <<<<<<<<<<<<<<<<<< Insert trained model name
apply_sigmoid = False                                                # <<<<<<<<<<<<<<<< Specify whether Sigmoid should
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
model_data = torch.load(trained_model_path)
model.load_state_dict(model_data['model'])
model.eval()

preprocessing = trainer.preprocessing
train_loader = dataloader.get_training_dataloader(split=1, batch_size=1, preprocessing=preprocessing)
test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

create_or_clean_directory(OUTPUT_PRED_DIR)

segmentation_threshold = compute_best_threshold(train_loader, apply_sigmoid=apply_sigmoid)

predict(segmentation_threshold, apply_sigmoid=apply_sigmoid)


