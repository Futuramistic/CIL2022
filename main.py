"""Main runner file. Looks for the "--model" or "-m" command line argument to determine the model to use,
then passes the remaining command line arguments to the constructor of the corresponding model's class."""

import argparse
import itertools
import re

from data_handling.dataloader_torch import TorchDataLoader
from data_handling.dataloader_tf import TFDataLoader
from models.road_extraction_from_high_res_rsi_using_dl.gl_dense_u_net import GLDenseUNet
from trainers.u_net import UNetTrainer
from trainers.gl_dense_u_net import GLDenseUNetTrainer

from utils import MODEL_CLASS_DICT

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
    model_class = MODEL_CLASS_DICT[known_args.model]
    model = model_class(**{k: v for k, v in arg_dict.items() if k.lower() not in filter_args})
    # TODO: train model, evaluate performance, ...
    
    # Testing the UNetTrainer
    # dataloader = TorchDataLoader(dataset='original')
    # trainer = UNetTrainer(dataloader, model, experiment_name='Vanilla UNet', evaluation_interval=1,
    #                       num_samples_to_visualize=3)
    # trainer.train()

    # model = GLDenseUNet(input_shape=[256, 256, 3])
    # dataloader = TFDataLoader(dataset='original_256')
    # trainer = GLDenseUNetTrainer(dataloader, model, experiment_name='Vanilla GLDenseUNet',
    #                              evaluation_interval=1, num_samples_to_visualize=3)
    # trainer.train()

except KeyError:
    print('Please specify a valid model using the "--model" parameter. Currently supported: ' +
          ', '.join(MODEL_CLASS_DICT.keys()))
