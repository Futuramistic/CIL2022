"""Main runner file. Looks for the "--model" or "-m" command line argument to determine the model to use,
then passes the remaining command line arguments to the constructor of the corresponding model's class."""

import argparse
import itertools
import re

from models.learning_aerial_image_segmenation_from_online_maps import Deeplabv3, Unet, Fastscnn
from models.road_extraction_from_high_res_rsi_using_dl.gl_dense_u_net import *

from data_handling.dataloader_torch import TorchDataLoader
from data_handling.dataloader_tf import TFDataLoader
from trainers.u_net import UNetTrainer
from trainers.gl_dense_u_net import GLDenseUNetTrainer

# dict of model names to corresponding classes
model_class_dict = {'deeplabv3': Deeplabv3.Deeplabv3,
                    'fastscnn': Fastscnn.FastSCNN,
                    'unet': Unet.UNet}
# list of arguments to avoid passing to constructor of model class
filter_args = ['h', 'model', 'm']

parser = argparse.ArgumentParser(description='Implementation of ETHZ CIL Road Segmentation 2022 project')
parser.add_argument('-m', '--model', type=str)
known_args, unknown_args = parser.parse_known_args()

remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
# float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
cast_arg = lambda s: int(s) if remove_leading_dashes(s).isdigit()\
                     else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None\
                     else bool(s) if s.lower() in ['true', 'false']\
                     else s

known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
unknown_args_dict = dict(map(lambda arg: (remove_leading_dashes(arg.split('=')[0]),
                                          cast_arg([*arg.split('='), True][1])),
                             unknown_args))
arg_dict = {**known_args_dict, **unknown_args_dict}

try:
    model_class = model_class_dict[known_args.model]
    model = model_class(**{k: v for k, v in arg_dict.items() if k.lower() not in filter_args})
    # TODO: train model, evaluate performance, ...
    
    # Testing the UNetTrainer
    # dataloader = TorchDataLoader(dataset='original')
    # trainer = UNetTrainer(dataloader, model, experiment_name='Vanilla UNet', evaluation_interval=1,
    #                       num_samples_to_visualize=3)
    # trainer.train()

except KeyError:
    print('Please specify a valid model using the "--model" parameter. Currently supported: ' +
          ', '.join(model_class_dict.keys()))
