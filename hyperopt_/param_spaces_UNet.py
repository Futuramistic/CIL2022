from hyperopt import hp
from hyperopt.pyll.base import scope
from utils import ROOT_DIR
import losses.precision_recall_f1 as f1

unet_1 = {
    'model': {
        'model_type': 'unet',
        'saving_directory': f"{ROOT_DIR}/archive/trials/UNet_2022_04_08-22_49",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {
            'n_channels': 3,
            'n_classes': 1
        }
    },
    'dataset': {
        'name': "original"
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            'split': hp.quniform('split', low=0.4, high=0.9, q=0.1), 
            'num_epochs': scope.int(hp.quniform('num_epochs', low = 1, high = 50, q=1)),
            'batch_size': scope.int(hp.quniform('batch_size', low = 1, high = 20, q=1)),
            'evaluation_interval': 1,
            'num_samples_to_visualize': 4, 
            'checkpoint_interval': 2
        }
    }
}

unet_2 = {
    'model': {
        'model_type': 'unettf',  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/trials/UNetTF_07_06_2022_0",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {
            'n_channels': 3,
            'n_classes': 1,
            'dropout': 0,
            'kernel_init': None,
            'kernel_regularizer': None,
            'use_learnable_pool': True,
            'up_transpose': hp.choice('up_transpose', [True, False]),
        },
    },
    'dataset': {
        'name': 'original',
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            # 'preprocessing': ...,  <-- currently not needed I guess
            'experiment_name': 'mlflow_test_exp', # optional
            'split': 0.9,
            'num_epochs': 80,
            'batch_size': scope.int(hp.quniform('batch_size', low=2, high=8, q=2)),
            'loss_function': 'FocalLoss',
            'num_samples_to_visualize': 16,
            'checkpoint_interval': 500,
            'loss_function_hyperparams': {
                'alpha': hp.quniform('alpha', low=0.1, high=0.9, q=0.2),
                'gamma': hp.quniform('gamma', low=1, high=5, q=1),
            },
        },
        # 'kwargs': {
        #     'loss_alpha': hp.quniform('loss_alpha', low=0.1, high=0.9, q=0.2),
        #     'loss_gamma': hp.quniform('loss_gamma', low=1, high=5, q=1),
        # }
    }
}