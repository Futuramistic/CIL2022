from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np
from utils import ROOT_DIR


deeplabv3_baseline_eval = {
    'model': {
        'model_type': 'deeplabv3',
        'saving_directory': f"{ROOT_DIR}/archive/models/deeplabv3",
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
            'experiment_name': 'DeepLabV3_Hyperopt',
            'split': 0.827,
            'num_epochs': 200,
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
deeplabv3_test = {
    'model': {
        'model_type': 'deeplabv3',
        'saving_directory': f"{ROOT_DIR}/archive/models/deeplabv3test",
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
            'split': 0.2,
            'num_epochs': 2,
            # scope.int: cast sampled value to integer
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(2), np.log(16), 2)),
            'checkpoint_interval': 100000,
            'hyper_seg_threshold': False,
            'blobs_removal_threshold': 0
        },
        'optimizer_params': {
            'optimizer_lr': hp.loguniform('learning_rate', np.log(5e-5), np.log(1e-2))
        }
    }
}