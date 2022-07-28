"""Main runner file. Looks for the "--model" or "-m" command line argument to determine the model to use,
then passes the remaining command line arguments to the constructor of the corresponding model's class."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import itertools
import json
import numpy as np
import random
import re
from adaboost.adaboost import AdaBooster

from factory import Factory
from utils import *
from utils.logging import pushbullet_logger

# keep these imports here so that we can access the corresponding namespaces from the "eval" call

import tensorflow as tf
import tensorflow.keras as K
import torch
import torch.nn
import torch.optim
import torch.nn.functional as F



def main():
    # all args that cannot be matched to the Trainer or DataLoader classes and are
    # not in filter_args will be passed to the
    # model's constructor

    trainer_args = ['experiment_name', 'E', 'run_name', 'R', 'split', 's', 'num_epochs', 'e', 'batch_size', 'b',
                    'optimizer_or_lr', 'l', 'loss_function', 'L', 'loss_function_hyperparams', 'H',
                    'evaluation_interval', 'i', 'num_samples_to_visualize', 'v', 'checkpoint_interval', 'c',
                    'load_checkpoint_path', 'C', 'segmentation_threshold', 't', 'use_channelwise_norm', 'history_size',
                    'max_rollout_len', 'std', 'reward_discount_factor', 'num_policy_epochs', 'policy_batch_size',
                    'sample_from_action_distributions', 'visualization_interval', 'min_steps', 'rollout_len',
                    'blobs_removal_threshold', 'T', 'hyper_seg_threshold', 'w', 'use_sample_weighting',
                    'use_adaboost', 'a', 'f1_threshold_to_log_checkpoint', 'deep_adaboost', 'D']
    dataloader_args = ['dataset', 'd', 'use_geometric_augmentation', 'use_color_augmentation',
                       'aug_brightness', 'aug_contrast', 'aug_saturation', 'use_adaboost', 'a']

    # list of other arguments to avoid passing to constructor of model class
    filter_args = ['h', 'model', 'm', 'evaluate', 'eval', 'V', 'seed', 'S', 'adaboost_runs', 'A']

    parser = argparse.ArgumentParser(description='Implementation of ETHZ CIL Road Segmentation 2022 project')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-S', '--seed', type=int, default=1, required=False)
    parser.add_argument('-E', '--experiment_name', type=str, required=True)
    parser.add_argument('-R', '--run_name', type=str, required=False)
    parser.add_argument('-s', '--split', type=float, default=DEFAULT_TRAIN_FRACTION, required=False)
    parser.add_argument('-e', '--num_epochs', type=int, required=False)
    parser.add_argument('-b', '--batch_size', type=int, required=False)
    parser.add_argument('-L', '--loss_function', type=str, required=False)
    # json.loads: substitute for dict
    parser.add_argument('-H', '--loss_function_hyperparams', type=json.loads, required=False, default=None)
    parser.add_argument('-i', '--evaluation_interval', type=float, required=False, default=10)
    parser.add_argument('-v', '--num_samples_to_visualize', type=int, required=False)
    parser.add_argument('-c', '--checkpoint_interval', type=int, required=False)
    parser.add_argument('-C', '--load_checkpoint_path', '--from_checkpoint', type=str, required=False)
    parser.add_argument('-t', '--segmentation_threshold', type=float, default=DEFAULT_SEGMENTATION_THRESHOLD,
                        required=False)
    parser.add_argument('-f', '--f1_threshold_to_log_checkpoint', type=float, default=DEFAULT_F1_THRESHOLD_TO_LOG_CHECKPOINT,
                        required=False)
    parser.add_argument('-B', '--blobs_removal_threshold', type=int, default=DEFAULT_BLOBS_REMOVAL_THRESHOLD,
                        required=False)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-V', '--evaluate', '--eval', action='store_true')
    parser.add_argument('-T', '--hyper_seg_threshold', type=bool,
                        help="If True, use hyperparameter search after evaluation to find the best "
                             "segmentation threshold", required=False, default=True)
    parser.add_argument('-w', '--use_sample_weighting', type=bool, required=False, default=False,
                        help="If True, use sample weighting during training to train more on samples with big errors.\
                            Currently only working with torch")
    parser.add_argument('-a', '--use_adaboost', type=bool, required=False, default=False,
                        help="If True, apply the Adaboost algorithm to the training (original or deep version)")
    parser.add_argument('-D', '--deep_adaboost', type=bool, required=False, default=False,
                        help="If True, apply the Deep Adaboost algorithm to the training (use_adaboost must be True)")
    parser.add_argument('-A', '--adaboost_runs', type=int, required=False, default=20,
                        help="Only if adaboost is used, specify the number of models to ensemble")
    known_args, unknown_args = parser.parse_known_args()

    # Process the commandline arguments
    remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
    # float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
    cast_arg = lambda s: s[1:-1] if s.startswith('"') and s.endswith('"') \
        else int(s) if remove_leading_dashes(s).isdigit() \
        else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None \
        else s.lower() == 'true' if s.lower() in ['true', 'false'] \
        else None if s.lower() == 'none' \
        else eval(s) if any([s.startswith('(') and s.endswith(')'),
                             s.startswith('[') and s.endswith(']'),
                             s.startswith('{') and s.endswith('}')]) \
        else s

    known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
    unknown_args_dict = dict(map(lambda arg: (remove_leading_dashes(arg.split('=')[0]),
                                              cast_arg([*arg.split('='), True][1])),
                                 unknown_args))
    arg_dict = {**known_args_dict, **unknown_args_dict}

    # seed everything

    random.seed(known_args.seed)
    torch.manual_seed(known_args.seed)
    np.random.seed(known_args.seed)
    tf.random.set_seed(known_args.seed)
    
    # Load the model by name from the factory
    factory = Factory.get_factory(known_args.model)
    
    # specify the arguments for the dataloader, trainer and model class
    model_spec_args = {k: v for k, v in arg_dict.items() if k.lower() not in [*trainer_args, *dataloader_args, *filter_args]}
    trainer_spec_args = {k: v for k, v in arg_dict.items() if k.lower() in trainer_args}
    dataloader_spec_args = {k: v for k, v in arg_dict.items() if k.lower() in dataloader_args}
    
    # call adaboost script if adaboost is used
    if known_args_dict["use_adaboost"]:
        adabooster = AdaBooster(factory, known_args_dict, unknown_args_dict, model_spec_args, trainer_spec_args,
                                dataloader_spec_args, known_args.deep_adaboost, IS_DEBUG)
        adabooster.run()
        return
    
    # Create the dataloader using the commandline arguments
    dataloader = factory.get_dataloader_class()(**dataloader_spec_args)
    
    # Create the model using the commandline arguments
    model = factory.get_model_class()(**model_spec_args)
    
    # Create the trainer using the commandline arguments
    trainer = factory.get_trainer_class()(dataloader=dataloader, model=model,
                                        **trainer_spec_args)
        

    # do not move these Pushbullet messages into the Trainer class, as this may lead to a large amount of
    # messages when using Hyperopt

    if ('evaluate' in known_args_dict and known_args_dict['evaluate']) or \
            ('eval' in known_args_dict and known_args_dict['eval']):
        # evaluate
        if trainer.load_checkpoint_path is not None:
            trainer._init_mlflow()
            trainer._load_checkpoint(trainer.load_checkpoint_path)
        metrics = trainer.eval()
        if not IS_DEBUG:
            pushbullet_logger.send_pushbullet_message(('Evaluation finished. Metrics: %s\n' % str(metrics)) + \
                                                    f'Hyperparameters:\n{trainer._get_hyperparams()}')
    else:
        # train
        if not IS_DEBUG:
            pushbullet_logger.send_pushbullet_message('Training started.\n' + \
                                                    f'Hyperparameters:\n{trainer._get_hyperparams()}')
        last_test_loss = trainer.train()
        if not IS_DEBUG:
            pushbullet_logger.send_pushbullet_message(('Training finished. Last test loss: %.4f\n' % last_test_loss) + \
                                                    f'Hyperparameters:\n{trainer._get_hyperparams()}')


if __name__ == '__main__':
    # Setup the stderr and stdout files
    abs_logging_dir = os.path.join(ROOT_DIR, LOGGING_DIR)
    if not os.path.isdir(abs_logging_dir):
        os.makedirs(abs_logging_dir)

    stderr_path = os.path.join(abs_logging_dir, f'stderr_{SESSION_ID}.log')
    stdout_path = os.path.join(abs_logging_dir, f'stdout_{SESSION_ID}.log')

    for path in [stderr_path, stdout_path]:
        if os.path.isfile(path):
            os.unlink(path)

    if IS_DEBUG:
        main()
    else:  # If not running in debug mode, redirect the stdout and stderr to some log files
        try:
            print(f'Session ID: {SESSION_ID}\n'
                  'Not running in debug mode\n'
                  'stderr and stdout will be written to "%s" and "%s", respectively\n' % (stderr_path, stdout_path))
            # buffering=1: use line-by-line buffering
            with open(stderr_path, 'w', buffering=1) as stderr_f, open(stdout_path, 'w', buffering=1) as stdout_f:
                with redirect_stderr(stderr_f), redirect_stdout(stdout_f):
                    main()
        except Exception as e:
            err_msg = f'*** Exception encountered: ***\n{e}'
            pushbullet_logger.send_pushbullet_message(err_msg)
            raise e
