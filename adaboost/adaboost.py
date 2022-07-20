from utils.logging import pushbullet_logger
from utils import ROOT_DIR
import os
import pickle
import numpy as np

class AdaBooster:
    def __init__(self, factory, known_args_dict, unknown_args_dict,  model_spec_args, trainer_spec_args, dataloader_spec_args, is_debug):
        """An Adaboost class for performing the classical adaboost algorithm
        Note that it is necessary to call each new adaboost experiment by a separate experiment name, otherwise the previous experiment
        will be loaded and continued

        Args:
            arguments are passed through the main function in main.py and were created from the user's inputs
        """
        self.factory = factory
        self.known_args_dict = known_args_dict
        self.unknown_args_dict = unknown_args_dict
        self.model_args = model_spec_args
        self.trainer_args = trainer_spec_args
        self.dataloader_args = dataloader_spec_args
        self.is_debug = is_debug
        
        # Create the dataloader using the commandline arguments
        self.dataloader = factory.get_dataloader_class()(**dataloader_spec_args)
        
        # load last adaboost setting if adaboost name already exists, otherwise create new file structure for current experiment
        self.checkpoint_paths = [] # checkpoint paths for best epoch of trained models
        self.experiment_names = [] # experiment names for trained models
        self.model_weights = [] # model weight (unnormalized) depending on the models overall performance
        # sample weights are managed by the dataloader in order to be able to return the trainer the needed data structures
        self.adaboost_run_path = os.path.join(ROOT_DIR, "/adaboost/adaboost_runs", self.known_args_dict['experiment_name'])
        if os.path.exists(self.adaboost_run_path):
            self.checkpoint_paths = pickle.load(open(os.path.join(self.adaboost_run_path, 'checkpoints.pkl', 'rb')))
            self.experiment_names = pickle.load(open(os.path.join(self.adaboost_run_path, 'experiments.pkl', 'rb')))
            self.model_weights = pickle.load(open(os.path.join(self.adaboost_run_path, 'model_weights.pkl', 'rb')))
            with open(os.path.join(self.adaboost_run_path, "data_weights.ply"), 'rb') as file:
                # data weights don't vary in size and are easier to work with as np arrays 
                self.dataloader.weights = np.load(file).astype(float)
        else:
            os.makedirs(self.adaboost_run_path)
            self.update_files()
    
    def evaluate(self):
        # evaluate
        for idx, checkpoint in enumerate(self.checkpoint_paths):
            self.trainer_args["load_checkpoint_path"] = checkpoint
            model = self.factory.get_model_class()(**self.model_args)
            self.trainer_args["experiment_name"] = self.experiment_names[idx]
            trainer = self.factory.get_trainer_class()(dataloader=self.dataloader, model=model,
                                                **self.trainer_args)
            trainer._init_mlflow()
            trainer._load_checkpoint(trainer.load_checkpoint_path)
            metrics = trainer.eval()
        if not self.is_debug:
            pushbullet_logger.send_pushbullet_message(('Evaluation finished. Metrics: %s\n' % str(metrics)) + \
                                                    f'Hyperparameters:\n{self.get_hyperparams()}')
    
    def train(self):
        # training loop
        curr_run_idx = len(self.experiment_names)
        while curr_run_idx < self.known_args_dict["adaboost_runs"]:
            
            # Create the model using the commandline arguments
            model = self.factory.get_model_class()(**self.model_args)
            
            # Create the trainer using the commandline arguments
            trainer = self.factory.get_trainer_class()(dataloader=self.dataloader, model=model,
                                                **self.trainer_args)
            
            # adapt the mlflow experiment name for each new model
            old_run_name = trainer.mlflow_experiment_name
            new_run_name = f"{str(old_run_name)}_run_{curr_run_idx}" if old_run_name is not None else f"Run_{curr_run_idx}"
            trainer.mlflow_experiment_name = new_run_name
            self.experiment_names.append(new_run_name)
            
            # TODO: trainer and dataloader both get adaboost info --> dataloader has to save the data weights whereas trainer has
            # to call specific functions from the dataloader, depending on torch or tensorflow
            # TODO: add argument to all trainers and dataloaders
            # TODO: update the data weights
            # TODO: calculate the model weights by evaluation score
            # TODO: append best checkpoint to the self.checkpoint list
            
            self.update_files() # save only if training went through
            curr_run_idx += 1
    
    def run(self):
        if ('evaluate' in self.known_args_dict and self.known_args_dict['evaluate']) or \
                        ('eval' in self.known_args_dict and self.known_args_dict['eval']):
            # evaluate
            self.evaluate()
        
        else:
            # train
            pushbullet_logger.send_pushbullet_message('Training started.\n' + \
                                                        f'Hyperparameters:\n{self.get_hyperparams()}')
            last_test_loss = self.train()
            
            pushbullet_logger.send_pushbullet_message(('Training finished. Last test loss: %.4f\n' % last_test_loss) + \
                                                        f'Hyperparameters:\n{self.get_hyperparams()}')
    
    def get_hyperparams(self):
        ...
    
    def update_files(self):
        pickle.dump(self.checkpoint_paths, open(os.path.join(self.adaboost_run_path, "checkpoints.pkl", 'wb')))
        pickle.dump(self.experiment_names, open(os.path.join(self.adaboost_run_path, "experiments.pkl", 'wb')))
        pickle.dump(self.model_weights, open(os.path.join(self.adaboost_run_path, "model_weights.pkl", 'wb')))
        with open(os.path.join(self.adaboost_run_path, "data_weights.ply"), 'rb') as file:
            np.save(os.path.join(self.adaboost_run_path, "data_weights.ply"), self.dataloader.weights)