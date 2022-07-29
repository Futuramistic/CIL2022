import cloudpickle
from functools import partial
from hyperopt import STATUS_OK, fmin, tpe, Trials, STATUS_FAIL

from data_handling import *
from factory import Factory, get_torch_scheduler
from utils.logging import pushbullet_logger

class HyperParamOptimizer:
    """Class for Hyperparameter Optimization, based on the F1 score of the validation split
    Args:
        param_space (Dictionary) - hyparparameter search space, beware that no two params are called the same,
        use the param_space_template.py to create new spaces
    """

    def __init__(self, param_space, param_space_name=None):
        """
        Args:
            param_space (dict): Dictionary describing the search space
            param_space_name (str, optional) : Name of the search space
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        self.param_space = param_space
        
        # get the corresponding model class, trainer class and a single dataloader instance
        factory = Factory.get_factory(param_space['model']['model_type'].lower())
        self.model_class = factory.get_model_class()
        self.trainer_class = factory.get_trainer_class()
        self.dataloader = (factory.get_dataloader_class()
                                                       (**{'dataset': param_space['dataset']['name'],
                                                        **param_space['dataset'].get('dataloader_params', {})}))
        
        # specify names for mlflow
        self.run_name = param_space_name
        if 'experiment_name' in param_space['training']['trainer_params']:
            self.exp_name = param_space['training']['trainer_params']['experiment_name']
            # delete to avoid error due to duplicate argument
            del param_space['training']['trainer_params']['experiment_name']
        else:
            self.exp_name = self.model_class.__name__
        
        # prepare file for saving trials and reload trials if file already exists        
        # code adapted from
        # https://stackoverflow.com/questions/63599879/can-we-save-the-result-of-the-hyperopt-trials-with-sparktrials
        self.trials_path = param_space['model']['saving_directory']
        if not os.path.isdir("archive"):
            os.makedirs("archive")
        if not os.path.isdir("archive/trials"):
            os.makedirs("archive/trials")
        if not os.path.isdir("archive/models"):
            os.makedirs("archive/models")
        if os.path.isdir(self.trials_path):
            with open(self.trials_path, 'rb') as file:
                self.trials = cloudpickle.load(file)
        else:
            open(self.trials_path, 'w')
            self.trials = Trials()
    
    def run(self, n_runs):
        """
        Executes search for best (hyper-)hyperparameters by training different models and
        computing the f1 score over the validation data
        The best model is chosen by the best f1 score
        Args: 
            n_runs (int) - number of searches in search space
        Returns:
            output (Hyperopt Trials): Trial history
        """

        # we log these Pushbullet messages regardless of whether we're in debug mode or not, as the optimization may
        # take very long
        pushbullet_logger.send_pushbullet_message("Hyperopt optimization started.")

        # run objective function n_runs times with different variations of the parameter space
        best_run = fmin(
                partial(self.objective),
                space = self.param_space,
                trials = self.trials,
                algo=tpe.suggest,
                max_evals = n_runs
            )
        
        print(f"-----------Best Run:-----------\n{best_run}")
        pushbullet_logger.send_pushbullet_message(f"Hyperopt optimization finished.\nBest run: {best_run}")
        return self.trials
    
    def objective(self, hyperparams):
        """
        The target function, which is to be minimized based on the 'loss' value in the return dictionary
        Args: 
            hyperparams (Dictionary) - Hyperparameter search space
        Returns:
            output (Dictionary[loss, status, other infos depending on result])
        """
        # for mlflow logger
        run_name_initial = self.run_name if self.run_name is not None else ''
        run_name = f"Hyperopt_{run_name_initial}" + "_{:%Y_%m_%d_%H_%M}".format(datetime.datetime.now())
        with open(self.trials_path, 'wb') as handle:
            cloudpickle.dump(self.trials, handle)
        
        # catch Exceptions during runtimes to keep the search alive, report failed runs
        try:
            training_params = hyperparams['training']['trainer_params']
            
            # for RL: Calculate input dimension depending on history size; check for reward restrictions
            if 'history_size' in training_params:
                hyperparams['model']['kwargs']['in_channels'] = \
                    int(training_params['history_size'] + 2 +
                        (1 if hyperparams['dataset']['name'] == 'original_gt' else 3))
                # penalty for false negative should not be smaller than reward for true positive,   
                training_params['rewards']['false_neg_seg_pen'] = training_params['rewards']['false_pos_seg_pen'] if \
                    training_params['rewards']['false_neg_seg_pen'] < training_params['rewards']['false_pos_seg_pen'] \
                    else training_params['rewards']['false_neg_seg_pen']
            
            # get model instance
            model = self.model_class(**hyperparams['model']['kwargs'])
            
            # get scheduler from string
            if 'optimizer_params' in hyperparams['training']:
                lr = hyperparams['training']['optimizer_params']['optimizer_lr']
                optimizer = self.trainer_class.get_default_optimizer_with_lr(lr, model)
                optimizer.lr = lr

                if 'scheduler_args' in hyperparams['training']['optimizer_params']:
                    scheduler_args = hyperparams['training']['optimizer_params']['scheduler_args']
                    scheduler_name = scheduler_args['scheduler_name']
                    scheduler_kwargs = scheduler_args['kwargs']
                    scheduler = get_torch_scheduler(optimizer, scheduler_name, scheduler_kwargs)
                    hyperparams['training']['trainer_params']['scheduler'] = scheduler
            else:
                optimizer = None
            
            # create trainer instance and train
            trainer = self.trainer_class(**training_params, dataloader=self.dataloader, model=model,
                                         experiment_name=self.exp_name, run_name=run_name,
                                         optimizer_or_lr=optimizer)
            trainer.train()
            
        except RuntimeError as r:
            err_msg = f"Hyperopt failed.\nCurrent hyperparams that lead to error:\n{hyperparams}" +\
                      f"\n\nError message:\n{r}"
            print(err_msg)
            pushbullet_logger.send_pushbullet_message(err_msg)
            return {
                'status': STATUS_FAIL,
                'params': hyperparams
            }
        
        # Successful run: Save (overwrite) updated trials after each training
        with open(self.trials_path, 'wb') as handle:
            cloudpickle.dump(self.trials, handle)
        return { # use inverse f1 score as loss because hyperopt tries to minimize the loss
            'loss': 1-abs(trainer.callback_handler.best_f1_score),
            'status': STATUS_OK,
            'trained_model': model,
            'params': hyperparams
        }
        
    @staticmethod
    def get_best_model(trials_path):
        """
        Returns the optimal model trained in previous trials under the trials_path (use ROOT_DIR)
        Args:
            trials_path (string or path) the absolute path to the trials pickle file
        Returns:
            the best model of all trials, measured by its validation f1 score
        """
        with open(trials_path, 'rb') as file:
            trials = cloudpickle.load(file)
        best_trial = HyperParamOptimizer.get_best_trial(trials)
        print(best_trial)
        return best_trial['result']['trained_model']
    
    @staticmethod
    def get_best_trial(trials):
        """
        Returns the experiment with the minimum f1 score
        Args:
            trial (hyperopt.Trials): Trials Object
        Returns:
            The trial with the best validation f1 score
        """
        # code adapted from
        # stackoverflow.com/questions/54273199/how-to-save-the-best-hyperopt-optimized-keras-models-and-its-weights
        valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
        losses = [float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minimum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minimum_loss]
        return best_trial_obj
