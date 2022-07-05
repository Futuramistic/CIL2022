from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np

from utils import *

simple_cnn_1 = {
    'model': {
        'model_type': "simplerlcnn",  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/simplrlcnn_hyperopt_2022_07_04",
        'kwargs': {
            'patch_size': hp.choice('patch_size', [[100,100], [50,50], [150, 150]]),
            'in_channels': 10
        }
    },
    'dataset': {
        'name': "original",
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'optimizer_params': {
            'scheduler_args': hp.choice('scheduler', [
                {'scheduler_name': 'steplr', 
                'kwargs':{
                    'step_size': scope.int(hp.quniform('step_size', low=20, high=100, q=10)),
                    'gamma': hp.uniform('gamma', low=1e-3, high=1e-1)
                    }
                },
                {'scheduler_name': 'lambdaevallr',
                'kwargs':{
                    'lr_lambda': hp.choice('lr_lambda', ['lambda epoch: 0.9*epoch', 'lambda epoch:0.99*epoch'])
                    }
                },
                {'scheduler_name':'plateau',
                'kwargs':{
                    'mode': 'min'
                    }
                }
            ]),
            'optimizer_lr':hp.uniform('optimizer_lr', low = 1e-5, high=1e-2)},
        'trainer_params':{
            'experiment_name': "Hyperopt_RL",
            'split': hp.choice('split', [0.4, 0.8]),
            'num_epochs': scope.int(hp.quniform('num_epochs', low = 50, high = 500, q=50)),
            'batch_size': scope.int(hp.quniform('batch_size', low = 1, high = 8, q=1)),
            'evaluation_interval': 2,
            'num_samples_to_visualize': 8,
            'checkpoint_interval': 25,
            # 'batch_size': , currently the batch size has no effect on the gradients
            'loss_function':None, 
            'loss_function_hyperparams':None, 
            'load_checkpoint_path': None,
            'segmentation_threshold':None, 
            'history_size': scope.int(hp.quniform('history_size', low = 5, high = 20, q=2)),
            'max_rollout_len':scope.int(hp.quniform('max_rollout_len', low = 16e4, high = 32e6, q=5e2)), 
            'replay_memory_capacity': scope.int(hp.quniform('replay_memory_capacity', low=1e3, high= 1e6, q=1e3)), 
            'std': hp.choice('std', [1e-3,
                                     1e-2,
                                     1e-1,
                                     [1e-2, 1e-2, 1e-2, 1e-1, 1e-3] #TODO
                                     ]),
            'reward_discount_factor': hp.uniform('reward_discount_factor', low=0.0, high=0.99),
            'num_policy_epochs': scope.int(hp.quniform('num_policy_epochs', low=5, high= 1e2, q=5)), 
            'policy_batch_size': scope.int(hp.quniform('policy_batch_size', low=16, high= 52, q=12)), 
            'sample_from_action_distributions': True, 
            'visualization_interval':20,
            'min_steps': scope.int(hp.quniform('min_steps', low=0, high= 1e4, q=10)),
            'rewards': {
                # penalty for false negative should not be smaller than reward for true positive,
                # else agent could paint a road spot, then erase it, then paint it again, etc.
                # (not observed, but possible loophole) --> is checked in hyperopt
                'changed_brush_pen': hp.uniform('changed_brush_pen', low=0.0, high=0.01),
                'changed_brush_rad_pen':hp.uniform('changed_brush_rad_pen', low=0.0, high=0.01),
                'changed_angle_pen': hp.uniform('changed_angle_pen', low=0.0, high=0.01),
                'false_neg_seg_pen': hp.uniform('false_neg_seg_pen', low=0.0, high=0.01),
                'false_pos_seg_pen': hp.uniform('false_pos_seg_pen', low=0.0, high=0.01),
                'time_pen': hp.uniform('time_pen', low=0.0, high=0.001),
                'unseen_pix_pen': hp.uniform('unseen_pix_pen', low=0.01, high=0.1),
                'true_pos_seg_rew': hp.uniform('true_pos_seg_rew', low=0.0, high=0.99),
                'true_neg_seg_rew': hp.uniform('true_neg_seg_rew', low=0.0, high=0.99),
                'unseen_pix_rew': hp.uniform('unseen_pix_rew', low=0.0, high=0.001)
            }
        }
    }
}

# reduced version of simple_cnn_1
simple_cnn_2 = {
    'model': {
        'model_type': "simplerlcnn",  # string, to search for the corresponding factory using factory.py
        'saving_directory': f"{ROOT_DIR}/archive/models/simplrlcnn_hyperopt_2022_07_04_2",
        'kwargs': {
            'patch_size': hp.choice('patch_size', [[100,100]]),
            'in_channels': 10
        }
    },
    'dataset': {
        'name': "original",
    },
    'training': {
        'minimize_loss': True, # always specify, as hyperopt can only minimize losses and therefore adapts the sign
        'optimizer_params': {
            'scheduler_args': hp.choice('scheduler', [
                {'scheduler_name': 'lambdaevallr',
                'kwargs':{
                    'lr_lambda': 'lambda epoch: 1.0'
                    }
                },
                {'scheduler_name':'plateau',
                'kwargs':{
                    'mode': 'min'
                    }
                }
            ]),
            'optimizer_lr':hp.uniform('optimizer_lr', low = 1e-5, high=1e-2)},
        'trainer_params': {
            'experiment_name': "Hyperopt_RL",
            'split': 0.5,
            'num_epochs': 10,
            'batch_size': 16,
            'evaluation_interval': 2,
            'num_samples_to_visualize': 9,
            'checkpoint_interval': 25,
            # 'batch_size': , currently the batch size has no effect on the gradients
            'loss_function':None, 
            'loss_function_hyperparams':None,
            'load_checkpoint_path': None,
            'segmentation_threshold':None, 
            'history_size': 5,
            'max_rollout_len': 500,  # keep it low, just to see if agent learns at all (agent should still be able to learn to do something useful during that time)
            'replay_memory_capacity': scope.int(hp.quniform('replay_memory_capacity', low=1e3, high= 1e6, q=1e3)), 
            'std': [0.02, 0.02, 0.1, 0.01, 0.4],
            'reward_discount_factor': hp.uniform('reward_discount_factor', low=0.0, high=0.99),
            'num_policy_epochs': scope.int(hp.quniform('num_policy_epochs', low=5, high=1e2, q=5)), 
            'policy_batch_size': scope.int(hp.quniform('policy_batch_size', low=16, high= 52, q=12)), 
            'sample_from_action_distributions': True, 
            'visualization_interval':20,
            'min_steps': 100,
            'rewards': {
                # penalty for false negative should not be smaller than reward for true positive,
                # else agent could paint a road spot, then erase it, then paint it again, etc.
                # (not observed, but possible loophole) --> is checked in hyperopt
                'changed_brush_pen': hp.uniform('changed_brush_pen', low=0.0, high=0.01),
                'changed_brush_rad_pen':hp.uniform('changed_brush_rad_pen', low=0.0, high=0.01),
                'changed_angle_pen': hp.uniform('changed_angle_pen', low=0.0, high=0.01),
                'false_neg_seg_pen': hp.uniform('false_neg_seg_pen', low=0.0, high=0.01),
                'false_pos_seg_pen': hp.uniform('false_pos_seg_pen', low=0.0, high=0.01),
                'time_pen': hp.uniform('time_pen', low=0.0, high=0.001),
                'unseen_pix_pen': hp.uniform('unseen_pix_pen', low=0.01, high=0.1),
                'true_pos_seg_rew': hp.uniform('true_pos_seg_rew', low=0.0, high=0.99),
                'true_neg_seg_rew': hp.uniform('true_neg_seg_rew', low=0.0, high=0.99),
                'unseen_pix_rew': hp.uniform('unseen_pix_rew', low=0.0, high=0.001)
            }
        }
    }
}