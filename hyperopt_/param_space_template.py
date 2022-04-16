from hyperopt import hp
from sklearn.metrics import f1_score
from models import UNet
from utils import ROOT_DIR

# For Tensorflow models (Trainer parameters differ between torch and tf)
space_for_tf = {
    'model': {
        'model_type': ...,  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/...",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {...
        }
    },
    'dataset': {
        'name': ...,
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            # 'preprocessing': ...,  <-- currently not needed I guess
            'steps_per_training_epoch': ...,
            'split': ..., 
            'num_epochs':...,
            'batch_size': ...,
            'optimizer_or_lr':..., 
            'loss_function':...,
            'evaluation_interval':...,
            'num_samples_to_visualize': ..., 
            'checkpoint_interval': ...
        }
    }
}

# For Torch Models
space_for_torch = {
    'model': {
        'model_type': ...,  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/...",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {...
        }
    },
    'dataset': {
        'name': ...,
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            # 'preprocessing': ...,  <-- currently not needed I guess
            'split': ..., 
            'num_epochs':...,
            'batch_size': ...,
            'optimizer':..., 
            'scheduler':...,
            'loss_function':...,
            'evaluation_interval':...,
            'num_samples_to_visualize': ..., 
            'checkpoint_interval': ...
        }
    }
}