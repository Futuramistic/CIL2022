from sklearn.preprocessing import MinMaxScaler
import torch
from lucent.optvis import render
from lucent.modelzoo import *
from lucent.modelzoo.util import get_model_layers
import argparse
from contextlib import redirect_stderr, redirect_stdout
import os
import warnings
import re
import hashlib
import pysftp
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
import sys
from PIL import Image
import numpy as np
import torchvision
from lucent.misc.io import show
import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
from lucent.misc.channel_reducer import ChannelReducer
from lucent.misc.io import show
from requests.auth import HTTPBasicAuth
from itertools import product

from utils import *
from utils.logging import pushbullet_logger
from factory import *
from utils.preprocessing import get_preprocessing

"""Script for visualizing the activation map of a neural network
"""


##################### Helper Functions ###########################
original_checkpoint_hash = None

def load_checkpoint_torch(checkpoint_path, model, device):
    print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
    load_from_sftp = checkpoint_path.lower().startswith('sftp://')
    if load_from_sftp:
        original_checkpoint_hash = hashlib.md5(str.encode(checkpoint_path)).hexdigest()
        final_checkpoint_path = f'original_checkpoint_{original_checkpoint_hash}.pt'
        if not os.path.isfile(final_checkpoint_path):
            print(f'Downloading checkpoint from "{checkpoint_path}" to "{final_checkpoint_path}"...')
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL, auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'], os.environ['MLFLOW_TRACKING_PASSWORD'])).text
            url_components = urlparse(checkpoint_path)
            with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_FTP_USER, password=mlflow_ftp_pass,
                                cnopts=cnopts) as sftp:
                sftp.get(url_components.path, final_checkpoint_path)
            print(f'Download successful')
        else:
            print(f'Checkpoint "{checkpoint_path}", to be downloaded to "{final_checkpoint_path}", found on disk')
    else:
        final_checkpoint_path = checkpoint_path

    print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
    checkpoint = torch.load(final_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('Checkpoint loaded\n')
    return model

def load_checkpoint_tf(checkpoint_path, model):
    print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
    load_from_sftp = checkpoint_path.lower().startswith('sftp://')
    if load_from_sftp:
        original_checkpoint_hash = hashlib.md5(str.encode(checkpoint_path)).hexdigest()
        # in TF, even though the checkpoint names all end in ".ckpt", they are actually directories
        # hence we have to use sftp_download_dir_portable to download them
        final_checkpoint_path = f'original_checkpoint_{original_checkpoint_hash}.ckpt'
        if not os.path.isdir(final_checkpoint_path):
            os.makedirs(final_checkpoint_path, exist_ok=True)
            print(f'Downloading checkpoint from "{checkpoint_path}" to "{final_checkpoint_path}"...')
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL,
                                        auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'],
                                                            os.environ['MLFLOW_TRACKING_PASSWORD'])).text
            url_components = urlparse(checkpoint_path)
            with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_FTP_USER, password=mlflow_ftp_pass,
                                cnopts=cnopts) as sftp:
                sftp_download_dir_portable(sftp, remote_dir=url_components.path, local_dir=final_checkpoint_path)
            print(f'Download successful')
        else:
            print(f'Checkpoint "{checkpoint_path}", to be downloaded to "{final_checkpoint_path}", found on disk')
    else:
        final_checkpoint_path = checkpoint_path

    print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
    model.load_weights(final_checkpoint_path)
    print('Checkpoint loaded\n')
    return model

def calculate_activations_torch(model, device, dataloader, preprocessing, layer):
    # model_layer = "model." + layer
    # model_layer = eval(model_layer)[int(layer_idx)]
    # code adapted from https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/activation_grids.ipynb#scrollTo=Z00-g_liJO19
    @torch.no_grad()
    def get_layer(X):
        layer_model = "model"
        skip_next = False
        for idx, name in enumerate(layer.split("_")):
            if skip_next:
                skip_next = False
                continue
            if idx == len(layer.split("_"))-1:
                layer_model = eval(layer_model)[int(name)]
            elif name == "double":
                name = name + "_" + layer.split("_")[idx+1]
                skip_next = True
                layer_model += "." + name
            else:
                layer_model += "." + name
        hook = render.ModuleHook(layer_model)
        model(X)
        hook.close()
        return hook.features
    
    def activation_grid(img, cell_image_size=60, n_groups=6, n_steps=10, batch_size=64):
        # First wee need, to normalize and resize the image
        img = img.to(device)
        transforms = transform.standard_transforms.copy() + [torch.nn.Upsample(size=224, mode="bilinear", align_corners=True)]
        transforms_f = transform.compose(transforms)
        # shape: (1, 3, original height of img, original width of img)
        img = img.unsqueeze(0)
        # shape: (1, 3, 224, 224)
        img = transforms_f(img)

        # Here we compute the activations of the layer `layer` using `img` as input
        # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
        acts = get_layer(img)[0]
        # shape: (layer_height, layer_width, layer_channels)
        acts = acts.permute(1, 2, 0)
        # shape: (layer_height*layer_width, layer_channels)
        acts = acts.view(-1, acts.shape[-1])
        acts_np = acts.cpu().numpy()
        nb_cells = acts.shape[0]

        # negative matrix factorization `NMF` is used to reduce the number
        # of channels to n_groups. This will be used as the following.
        # Each cell image in the grid is decomposed into a sum of
        # (n_groups+1) images. First, each cell has its own set of parameters
        #  this is what is called `cells_params` (see below). At the same time, we have
        # a of group of images of size 'n_groups', which also have their own image parametrized
        # by `groups_params`. The resulting image for a given cell in the grid
        # is the sum of its own image (parametrized by `cells_params`)
        # plus a weighted sum of the images of the group. Each each image from the group
        # is weighted by `groups[cell_index, group_idx]`. Basically, this is a way of having
        # the possibility to make cells with similar activations have a similar image, because
        # cells with similar activations will have a similar weighting for the elements
        # of the group.
        if n_groups > 0:
            reducer = ChannelReducer(n_groups, "NMF")
            scaler = MinMaxScaler()
            groups = reducer.fit_transform(scaler.fit_transform(acts_np))
            groups /= groups.max(0)
        else:
            groups = np.zeros([])
        # shape: (layer_height*layer_width, n_groups)
        groups = torch.from_numpy(groups)
        # Parametrization of the images of the groups (we have 'n_groups' groups)
        groups_params, groups_image_f = param.fft_image([n_groups, 3, cell_image_size, cell_image_size])
        # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
        cells_params, cells_image_f = param.fft_image(
            [nb_cells, 3, cell_image_size, cell_image_size]
        )

        # First, we need to construct the images of the grid from the parameterizations
        def image_f():
            groups_images = groups_image_f()
            cells_images = cells_image_f()
            X = []
            for i in range(nb_cells):
                x = 0.7 * cells_images[i] + 0.5 * sum(groups[i, j] * groups_images[j] for j in range(n_groups))
                X.append(x)
            X = torch.stack(X)
            return X

        # make sure the images are between 0 and 1
        image_f = param.to_valid_rgb(image_f, decorrelate=True)
        # After constructing the cells images, we sample randomly a mini-batch of cells
        # from the grid. This is to prevent memory overflow, especially if the grid
        # is large.
        def sample(image_f, batch_size):
            def f():
                X = image_f()
                inds = torch.randint(0, len(X), size=(batch_size,))
                inputs = X[inds]
                # HACK to store indices of the mini-batch, because we need them
                # in objective func. Might be better ways to do that
                sample.inds = inds
                return inputs
            return f

        image_f_sampled = sample(image_f, batch_size=batch_size)
        # Now, we define the objective function

        def objective_func(model_func):
            # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
            pred = model_func(layer)
            # use the sampled indices from `sample` to get the corresponding targets
            target = acts[sample.inds].to(device)
            # shape: (batch_size, layer_channels, 1, 1)
            target = target.view(target.shape[0], target.shape[1], 1, 1)
            dot = (pred * target).sum(dim=1).mean()
            return -dot

        obj = objectives.Objective(objective_func)

        def param_f():
            # We optimize the parametrizations of both the groups and the cells
            params = list(groups_params) + list(cells_params)
            return params, image_f_sampled
        print(cell_image_size)
        results = render.render_vis(model, obj, param_f, thresholds=(n_steps,), show_image=True, progress=True, fixed_image_size=cell_image_size)
        # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
        imgs = image_f()
        imgs = imgs.cpu().data
        imgs = imgs[:, :, 2:-2, 2:-2]
        # turn imgs into a a grid
        grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)), padding=0)
        grid = grid.permute(1, 2, 0)
        grid = grid.numpy()
        render.show(grid)
        return imgs
    
    images = dataloader.get_training_dataloader(split=0.1, batch_size=1,
                                                                    preprocessing=preprocessing)
    for (image,y) in images:
        activation_grid(image[0]) # reduce one dimension, batch size is 1 anyways

######################### main ##########################
def main_model_visualizer():
    warnings.filterwarnings("ignore", category=UserWarning) 
    # list of supported arguments
    filter_args = ['h', 'm', 'model', 'c', 'checkpoint_path', 'p', 'preprocessing', 'dataset', 'd']
    dataloader_args = ['dataset', 'd']

    parser = argparse.ArgumentParser(description='Implementation of Visualizer for the activation map of a Neural Network')
    parser.add_argument('-m', '--model', type=str, required=True, help="type of the model")
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help="model param path starting with 'sftp://' or locally at project root, e.g. 'checkpoints/165782452/cp_best_f1.pt'")
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-p', '--preprocessing', type=str, default="channelwise_preprocessing", help="name of preprocessor as stated in getter function from utils/preprocessing.py")
    known_args, unknown_args = parser.parse_known_args()
    remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
    # float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
    cast_arg = lambda s: s[1:-1] if s.startswith('"') and s.endswith('"')\
                         else int(s) if remove_leading_dashes(s).isdigit()\
                         else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None\
                         else s.lower() == 'true' if s.lower() in ['true', 'false']\
                         else None if s.lower() == 'none'\
                         else eval(s) if any([s.startswith('(') and s.endswith(')'),
                                              s.startswith('[') and s.endswith(']'),
                                              s.startswith('{') and s.endswith('}')])\
                         else s

    known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
    unknown_args_dict = dict(map(lambda arg: (remove_leading_dashes(arg.split('=')[0]),
                                            cast_arg([*arg.split('='), True][1])),
                                unknown_args))
    arg_dict = {**known_args_dict, **unknown_args_dict}

    factory = Factory.get_factory(known_args.model)
    dataloader = factory.get_dataloader_class()(**{k: v for k, v in arg_dict.items() if k.lower() in dataloader_args})
    is_torch = isinstance(dataloader, TorchDataLoader)
    model = factory.get_model_class()(**{k: v for k, v in arg_dict.items() if k.lower() not in [*filter_args]})
    preprocessing = get_preprocessing(known_args.preprocessing, dataloader)
    if is_torch:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device) # must be called before loading parameters into model
        model = load_checkpoint_torch(known_args.checkpoint_path, model, device)
        model.eval()
        print("Layer names:")
        for name in get_model_layers(model):
            print(name)
        layer = input(">>>>>>>>>>>>>>>>> Please Enter the name of the layer you want to visualize:\n")
        images = calculate_activations_torch(model, device, dataloader, preprocessing, layer)
        
    else:
        model = load_checkpoint_tf(known_args.checkpoint_path, model)
    render.render_vis(model, "mixed4a:476")


if __name__ == '__main__':
    main_model_visualizer()
