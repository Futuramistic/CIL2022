from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np

from utils import *


# TODO: find good parameter spaces; these parameter spaces are by no means good, they just work reasonably well to test GLDenseUNet with Hyperopt

GLDenseUNet_1 = {
    'model': {
        'model_type': 'gldenseunet',  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/trials/GLDenseUNet_2022_04_16-20_35",
        # use kwargs for class-specific parameters, as hyperopt is written generically
        'kwargs': {
            'input_shape': (256, 256, 3)
        }
    },
    'dataset': {
        'name': 'original_256',
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'trainer_params': {
            # 'preprocessing': ...,  <-- currently not needed I guess
            'split': 0.97,  # this high value is to compensate for the immense size of the original_128 and original_256 datasets
            'num_epochs': 1,

            # batch size 16 does not fit into 24GB of VRAM (could try 8 < B < 16; higher B is faster)
            'batch_size': scope.int(hp.choice('batch_size', [1, 2, 4, 8])),

            # log(optimizer_or_lr) is uniform if optimizer_or_lr is log-uniform; optimizer_or_lr constrained to [e^[1], e^[2]]
            # idea: try hp.choice first, then refine LR using lognormal based on which LR worked best
            # see what the distribution would look like on
            # https://colab.research.google.com/drive/1fzasOa4u0gZxpmWtLKe81Xf2bTJdvwee?usp=sharing
            # especially check if presumably desirable LRs are assigned a reasonably high probabilistic mass
            # hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2))
            'optimizer_or_lr': scope.float(hp.choice('optimizer_or_lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])),
            
            'evaluation_interval': 150,
            'num_samples_to_visualize': 4, 
            'checkpoint_interval': 1e15,  # only create a checkpoint after training
            'segmentation_threshold': 0.5
        }
    }
}