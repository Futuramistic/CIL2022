"""Main runner file. Looks for the "--model" or "-m" command line argument to determine the model to use,
then passes the remaining command line arguments to the constructor of the corresponding model's class."""

import argparse
import itertools
import re

from data_handling.dataloader_torch import TorchDataLoader
from data_handling.dataloader_tf import TFDataLoader
from factory.factory import Factory
from trainers.u_net import UNetTrainer
from trainers.gl_dense_u_net import GLDenseUNetTrainer
from utils import *


# all args that cannot be matched to the Trainer or DataLoader classes and are not in filter_args will be passed to the
# model's constructor

trainer_args = ['experiment_name', 'E', 'run_name', 'R', 'split', 's', 'num_epochs', 'e', 'batch_size', 'b',
                'optimizer_or_lr', 'l', 'evaluation_interval', 'i',
                'num_samples_to_visualize', 'v', 'checkpoint_interval', 'c', 'segmentation_threshold', 't']
dataloader_args = ['dataset', 'd']

# list of other arguments to avoid passing to constructor of model class
filter_args = ['h', 'model', 'm']

parser = argparse.ArgumentParser(description='Implementation of ETHZ CIL Road Segmentation 2022 project')
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-E', '--experiment_name', type=str, required=True)
parser.add_argument('-R', '--run_name', type=str, required=False)
parser.add_argument('-s', '--split', type=float, default=DEFAULT_TRAIN_FRACTION, required=False)
parser.add_argument('-e', '--num_epochs', type=int, required=False)
parser.add_argument('-b', '--batch_size', type=int, required=False)
parser.add_argument('-l', '--optimizer_or_lr', type=float, required=False)
parser.add_argument('-i', '--evaluation_interval', type=float, required=False)
parser.add_argument('-v', '--num_samples_to_visualize', type=int, required=False)
parser.add_argument('-c', '--checkpoint_interval', type=int, required=False)
parser.add_argument('-t', '--segmentation_threshold', type=float, default=DEFAULT_SEGMENTATION_THRESHOLD, required=False)
parser.add_argument('-d', '--dataset', type=str, required=True)
known_args, unknown_args = parser.parse_known_args()

remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
# float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
cast_arg = lambda s: int(s) if remove_leading_dashes(s).isdigit()\
                     else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None\
                     else bool(s) if s.lower() in ['true', 'false']\
                     else eval(s) if (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']'))\
                     else s

known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
unknown_args_dict = dict(map(lambda arg: (remove_leading_dashes(arg.split('=')[0]),
                                          cast_arg([*arg.split('='), True][1])),
                             unknown_args))
arg_dict = {**known_args_dict, **unknown_args_dict}

factory = Factory.get_factory(known_args.model)
dataloader = factory.get_dataloader_class()(**{k: v for k, v in arg_dict.items() if k.lower() in dataloader_args})
model = factory.get_model_class()(**{k: v for k, v in arg_dict.items() if k.lower() not in [*trainer_args,
                                                                                            *dataloader_args,
                                                                                            *filter_args]})
trainer = factory.get_trainer_class()(dataloader=dataloader, model=model,
                                      **{k: v for k, v in arg_dict.items() if k.lower() in trainer_args})
trainer.train()
