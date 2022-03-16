from functools import partial
from hyperopt import STATUS_OK, fmin, tpe, Trials
import pickle
import os
import numpy as np
from data_handling.dataloader_torch import TorchDataLoader

class HyperParamOptimizer:
    """Class for Hyperparameter Optimization
        Args: 
            param_space (Dictionary) - hyparparameter search space, beware that no two params are called the same, see param_spaces_UNet.py for an example
    """
    def __init__(self, param_space):
        # set often used parameters
        self.param_space = param_space
        self.model_class = param_space['model']['class']
        # create dataloader which is used for all runs
        self.dataloader = TorchDataLoader(param_space['dataset']['name']) if param_space['model']['is_torch_type'] else None #TODO: Change None to TFDataLoader when implemented
        
        # hyperopt can only minimize loss functions --> change sign if loss has to be maximized
        self.minimize_loss = param_space['training']['minimize_loss']
        
        # prepare file for saving trials and eventually reload trials if file already exists
        # code adapted from https://stackoverflow.com/questions/63599879/can-we-save-the-result-of-the-hyperopt-trials-with-sparktrials
        self.trials_path = param_space['model']['saving_directory']
        if os.path.isdir(self.trials_path):
            with open(self.trials_path, 'rb') as file:
                self.trials = pickle.load(file)
        else:
            self.trials = Trials()
    
    def run(self, n_runs):
        """
        Executes search for best (hyper-)hyperparameters by training different models and computing the loss over the validation data
        The best model is defined by the best validation loss!
        Args: 
            n_runs (int) - number of searches in search space
        Returns:
            output (Hyperopt Trials): Trial history
        """
        # run objective function n times with different variations of the parameter space and search for the variation minimizing the loss
        best_run = fmin(
                partial(self.objective),
                space = self.param_space,
                trials = self.trials,
                algo=tpe.suggest,
                max_evals = n_runs
            )
        print(f"------------Best Run:-----------\n{best_run}")
        
        return self.trials
    
    def objective(self, hyperparams):
        """
        The target function, which is to be minimized based on the 'loss' in the return dictionary
        Args: 
            hyperparams (Dictionary) - Hyperparameter search space
        Returns:
        output (Dictionary[loss, status, and other things])
        """
        print(self.model_class)
        model = self.model_class(**hyperparams['model']['kwargs'])
        
        # use Trainer here
        trainer = ...
        # Trainer(dataLoader = self.dataloader, model = model, **hyperparams['training']['trainer_params'])
        train_loss = trainer.train()
        test_loss = trainer.test()
        
        # save (overwrite) updated trials after each trainig
        with open(self.trials_path, 'r+') as handle:
            handle.truncate(0)
            pickle.dump(
                self.trials, handle)
        return {
            'loss': test_loss if self.minimize_loss else -test_loss,
            'status': STATUS_OK,
            'trained_model': model,
            'test_loss': test_loss,
            'train_loss': train_loss,
            'params': hyperparams
        }
    
    def get_best_model(self, trials_path = None):
        """
        Returns the optimal model trained in previous trials under the trials_path (use ROOT_DIR)
        """
        if trials_path is None:
            with open(self.trials_path, 'rb') as file:
                    trials = pickle.load(file)
        else:
            with open(trials_path, 'rb') as file:
                    trials = pickle.load(file)
        model = self.__get_best_model(trials)
        return model
    
    def __get_best_trial(trials):
        # code adapted from https://stackoverflow.com/questions/54273199/how-to-save-the-best-hyperopt-optimized-keras-models-and-its-weights
        valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
        losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        return best_trial_obj
    
    def __get_best_model(self, trials):
        best_trial = self.__get_best_trial(trials)
        return best_trial['result']['trained_model']