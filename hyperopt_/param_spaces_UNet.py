from hyperopt import hp
from hyperopt.pyll.base import scope
from utils import ROOT_DIR
import losses.f1 as f1
unet_1 = {
    'model': {
        'model_type': 'unet',
        'saving_directory': f"{ROOT_DIR}/archive/trials/UNet_2022-03-16_21_25",
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
            'split': 0.8, 
            'num_epochs': 1, # test run
            'batch_size': scope.int(hp.quniform('batch_size', low = 1, high = 10, q=1)),
            'evaluation_interval': 1,
            'num_samples_to_visualize': 4, 
            'checkpoint_interval': 2
        }
    }
}