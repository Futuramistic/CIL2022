from hyperopt import hp
from sklearn.metrics import f1_score
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from utils import ROOT_DIR

# For Tensorflow models
space_for_tf = {
    'model': {
        'class': ...,
        'is_torch_type': False, # always specify torch or tensor
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
            'preprocessing': ...,
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
        'class': ...,
        'is_torch_type': True, # always specify torch or tensor
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
            'preprocessing': ...,
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