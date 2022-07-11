from functools import partial
from hyperopt import STATUS_OK, fmin, tpe, Trials, STATUS_FAIL
from hyperopt import hp
import warnings

from data_handling import *
from utils.logging import pushbullet_logger


class ThresholdOptimizer:
    """Class for segmentation threshold optimization, based on the F1 score
    """
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.trials = Trials()
        self.n_runs = 15
        self.space = {
            "threshold": hp.uniform("threshold", low=0.0, high=1.0)
        }
    
    def run(self, predictions, targets, f1_score_function):
        """
        Executes search for best (hyper-)hyperparameters by training different models and computing the loss over the validation data
        The best model is defined by the best validation loss!
        Args: 
            predictions (tensors) - model's predictions for the evaluation data
        Returns:
            output (Hyperopt Trials): Trial history
        """
        self.prediction = predictions
        self.target = targets
        self.f1_score_function = f1_score_function

        # run objective function n times with different variations of the parameter space and search for the variation minimizing the loss
        best_run = fmin(
                partial(self.objective),
                space = self.space,
                trials = self.trials,
                algo=tpe.suggest,
                max_evals = self.n_runs,
                verbose=False
            )
        
        return self.get_best_threshold()
    
    def objective(self, hyperparams):
        """
        The target function, which is to be minimized based on the 'loss' in the return dictionary
        Args: 
            hyperparams (Dictionary) - search space
        Returns:
        output (Dictionary[loss, status, and other things])
        """
        try:
            f1_scores = []
            for idx, pred in enumerate(self.prediction):
                target = self.target[idx]
                thresholded_predictions = pred >= hyperparams["threshold"]
                f1_scores.append(self.f1_score_function(thresholded_predictions, target))
            average_f1_score = np.mean(f1_scores)
        except RuntimeError as r:
            err_msg = f"Hyperopt failed.\nCurrent hyperparams that lead to error:\n{hyperparams}" +\
                      f"\n\nError message:\n{r}"
            print(err_msg)
            pushbullet_logger.send_pushbullet_message(err_msg)
            return {
                'status': STATUS_FAIL,
                'params': hyperparams
            }
        return {
            'loss': 1-average_f1_score, 
            'status': STATUS_OK,
            'params': hyperparams
        }
    
    def get_best_threshold(self):
        # code adapted from https://stackoverflow.com/questions/54273199/how-to-save-the-best-hyperopt-optimized-keras-models-and-its-weights
        valid_trial_list = [trial for trial in self.trials if STATUS_OK == trial['result']['status']]
        losses = [float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        return float(best_trial_obj["result"]["params"]["threshold"])