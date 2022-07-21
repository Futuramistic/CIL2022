from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np
from utils import ROOT_DIR


segformer_baseline_eval = {
    'model': {
        'model_type': 'segformer',
        'saving_directory': f"{ROOT_DIR}/archive/models/segformer",
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
        'minimize_loss': True,  # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params': {
            'experiment_name': 'SegFormer_Hyperopt',
            'split': 0.827,
            'num_epochs': 600,
            # scope.int: cast sampled value to integer
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(2), np.log(4), 2)),
            'checkpoint_interval': 100000,
            'hyper_seg_threshold': True,
            'blobs_removal_threshold': 0
        },
        'optimizer_params': {
            'optimizer_lr': hp.loguniform('learning_rate', np.log(1e-6), np.log(5e-4))
        }
    }
}
