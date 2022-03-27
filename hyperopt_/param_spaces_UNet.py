from hyperopt import hp
from sklearn.metrics import f1_score
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from utils import ROOT_DIR
space_Unet_1 = {
    'model': {
        'class': UNet,
        'is_torch_type': True, # always specify: "torch" or "tensor"
        'saving_directory': f"{ROOT_DIR}/archive/models/UNet_2022-03-16_21_25",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {
            'n_channels': 3,
            'n_classes': 2
        }
    },
    'optimizer': {
        'method': 'Adam',
        'kwargs': { # always use kwargs for class/method-specific parameters, as hyperopt is written generically
            'lr': hp.uniform('lr', low=1e-5, high=1e-1),
            'lr_decay': hp.quniform('lr_decay', low=100, high=1000, q=1)
        }
    },
    'dataset': {
        'name': "original",
        'splitting_ratio': hp.uniform('splitting_ratio', low=0.5, high = 1.0)
    },
    'training': {
        'minimize_loss': True,
        'trainer_params':{
            'loss': f1_score,
            'num_epochs': 5,
            'batch_size': hp.quniform('batch_size', low = 1, high = 40, q = 1)
        }
    }
}