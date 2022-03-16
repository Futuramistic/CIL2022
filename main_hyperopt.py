"""Hyperopt runner file. Looks for the "--search_space" or "-s" command line argument to determine the feature_space to use,
then passes the remaining command line arguments to a Hyperparameter optimizer."""

import argparse
from hyperopt_.HyperOptimizer import HyperParamOptimizer
from hyperopt_.param_spaces_UNet import *


# list of arguments to avoid passing to constructor of model class
filter_args = ['h', 'search_space', 's', 'num_runs', 'n']

parser = argparse.ArgumentParser(description='Implementation of Hyperparameter Optimizer that searches through the parameter space')
parser.add_argument('-s', '--search_space', type=str)
parser.add_argument('-n', '--num_runs', type = int)
params = parser.parse_args()

optimizer = HyperParamOptimizer(eval(params.search_space))
optimizer.run(n_runs=params.num_runs)
