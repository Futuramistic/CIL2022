from utils import ROOT_DIR
from hyperopt.pyll.base import scope
from hyperopt import hp

attunet_1 = {
    'model': {
        'model_type': "attunet",  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/attunet_hyperopt_2022_04_09",
        'kwargs': {
            'input_shape': (400,400,3)
        }
    },
    'dataset': {
        'name': "original",
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            #'preprocessing': None,
            #'steps_per_training_epoch': scope.int(hp.quniform('steps_per_training_epoch', low=1, high=10, q=1)),
            'split': 0.1,
            #'split': hp.quniform('split', low=0.4, high=0.9, q=0.1),
            'num_epochs':1,
            #'num_epochs': scope.int(hp.quniform('num_epochs', low = 1, high = 50, q=1)),
            'batch_size': scope.int(hp.quniform('batch_size', low = 1, high = 8, q=1)),
            'evaluation_interval':1,
            'num_samples_to_visualize': 4,
            'checkpoint_interval': 1
        }
    }
}