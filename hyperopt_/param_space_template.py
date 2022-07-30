from utils import ROOT_DIR

"""
This file contains templates for Tensorflow and Torch search parameter spaces
"""

# For Tensorflow models (Trainer parameters differ between torch and tf)
space_for_tf = {
    'model': {
        'model_type': ...,  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/...",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': { ... }
    },
    'dataset': {
        'name': ...,
        'dataloader_params': { ... }
    },
    'training': {
        'trainer_params': {
            'experiment_name': ...,  # optional
            'split': ...,
            'num_epochs': ...,
            'batch_size': ...,
            'optimizer_or_lr': ...,
            'loss_function': ...,
            'loss_function_hyperparams': ...,
            'evaluation_interval': ...,
            'num_samples_to_visualize': ...,
            'checkpoint_interval': ...,
            'segmentation_threshold': ...
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
        'dataloader_params': { ... }
    },
    'training': {
        'trainer_params': {
            'experiment_name': ...,  # optional
            'split': ...,
            'num_epochs': ...,
            'batch_size': ...,
            'optimizer': ...,
            'scheduler': ...,
            'loss_function': ...,
            'evaluation_interval': ...,
            'num_samples_to_visualize': ...,
            'checkpoint_interval': ...,
            'segmentation_threshold': ...
        }
    }
}
