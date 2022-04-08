from hyperopt import hp
from hyperopt.pyll.base import scope
from utils import ROOT_DIR
import losses.f1 as f1
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