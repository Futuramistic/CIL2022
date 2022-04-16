"""Hyperopt runner file. Looks for the "--search_space" or "-s" command line argument to determine the feature_space to use,
then passes the remaining command line arguments to a Hyperparameter optimizer."""

import argparse
from hyperopt_.HyperOptimizer import HyperParamOptimizer
from hyperopt_ import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 
# list of supported arguments
filter_args = ['h', 'search_space', 's', 'num_runs', 'n']

parser = argparse.ArgumentParser(description='Implementation of Hyperparameter Optimizer that searches through the parameter space')
parser.add_argument('-s', '--search_space', type=str, help="Give the name of the search space, which has to be defined in 'hyperopt\\param_space_...' and imported in main_hyperopt")
parser.add_argument('-n', '--num_runs', type = int, help="The number of Hyperopt runs, aka models trained with different parameters from the search space")
params = parser.parse_args()

try:
    optimizer = HyperParamOptimizer(eval(params.search_space))
    optimizer.run(n_runs=params.num_runs)
except NameError as n:
    print(f"Check if you wrote the search space name correctly. Error message:\n{n}")
