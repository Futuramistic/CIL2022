import numpy as np

from hyperopt import hp
from hyperopt.pyll.base import scope
from utils import ROOT_DIR


cranet_baseline_eval = {
    'model': {
        'model_type': 'cranet',
        'saving_directory': f"{ROOT_DIR}/archive/models/cranet",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': { }
    },
    'dataset': {
        'name': 'original_split_1',
        'dataloader_params': {
            'use_geometric_augmentation': True,
            'use_color_augmentation': True
        }
    },
    'training': {
        'trainer_params': {
            'experiment_name': 'CRANet_Hyperopt',
            'split': 0.827,
            'num_epochs': 300,
            # scope.int: cast sampled value to integer
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(2), np.log(16), 2)),
            'checkpoint_interval': 100000,
            'hyper_seg_threshold': True,
            'blobs_removal_threshold': 0
        },
        'optimizer_params': {
            'optimizer_lr': hp.loguniform('learning_rate', np.log(5e-5), np.log(1e-2))
        }
    }
}

# Test search space
cranet_test = {
    'model': {
        'model_type': 'cranet',
        'saving_directory': f"{ROOT_DIR}/archive/models/cranettest",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': { }
    },
    'dataset': {
        'name': 'original',
        'dataloader_params': {
            'use_geometric_augmentation': False,
            'use_color_augmentation': False
        }
    },
    'training': {
        'trainer_params': {
            'experiment_name': 'CodeTesting',
            'split': 0.02,
            'num_epochs': 2,
            'batch_size': 1,
            'checkpoint_interval': 100000,
            'hyper_seg_threshold': False,
            'blobs_removal_threshold': 0
        },
        'optimizer_params': {
            'optimizer_lr': hp.loguniform('learning_rate', np.log(5e-5), np.log(1e-2))
        }
    }
}