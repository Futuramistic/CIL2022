from hyperopt import hp
from sklearn.metrics import f1_score
from utils import ROOT_DIR
unet_1 = {
    'model': {
        'model_type': 'unet',
        'saving_directory': f"{ROOT_DIR}/archive/models/UNet_2022-03-16_21_25",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {
            'n_channels': 3,
            'n_classes': 2
        }
    },
    'dataset': {
        'name': "original"
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params':{
            'split': 0.9, 
            'num_epochs': 1, # test run
            'batch_size': hp.quniform('batch_size', low = 5, high = 20, q=1),
            'evaluation_interval': 1,
            'num_samples_to_visualize': 4, 
            'checkpoint_interval': 2
        }
    }
}