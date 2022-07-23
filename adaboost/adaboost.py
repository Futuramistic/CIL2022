from trainers.trainer_torch import TorchTrainer
from utils.logging import pushbullet_logger
from utils import ROOT_DIR, OUTPUT_PRED_DIR, create_or_clean_directory
from torch_predictor import compute_best_threshold
import mask_to_submission

import os
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

class AdaBooster:
    def __init__(self, factory, known_args_dict, unknown_args_dict,  model_spec_args, trainer_spec_args, dataloader_spec_args, is_debug):
        """An Adaboost class for performing the classical adaboost algorithm. The evaluation parameter will create the submission files on
        the test data set.
        Note that it is necessary to call each new adaboost experiment by a separate experiment name, otherwise the previous experiment
        will be loaded and continued

        Args:
            arguments are passed through the main function in main.py and were created from the user's inputs
            for evaluation, the argument '--apply_sigmoid' (bool) is required, and if torch is used, additionally '--blob_threshold' (int)
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
        self.test_dataloader = factory.get_dataloader_class()(dataset="test")
        
        # load last adaboost setting if adaboost name already exists, otherwise create new file structure for current experiment
        self.checkpoint_paths = [] # checkpoint paths for best epoch of trained models
        self.experiment_names = [] # experiment names for trained models
        self.model_weights = [] # model weight (unnormalized) depending on the models overall performance
        # sample weights are managed by the dataloader in order to be able to return the trainer the needed data structures
        self.adaboost_run_path = os.path.join(ROOT_DIR, "/adaboost/adaboost_runs", self.known_args_dict['experiment_name'])
        self.submission_folder = os.path.join(self.adaboost_run_path, "submission")
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
    
    def evaluate(self):
        # creates single submission and calls ensembled submission
        os.makedirs(self.submission_folder)
        for idx, checkpoint in enumerate(self.checkpoint_paths):
            exp_name = self.experiment_names[idx]
            trainer_class = self.factory.get_trainer_class()
            
            # create predictions on test data set
            if issubclass(trainer_class, TorchTrainer):
                # Torch
                os.system('python', 'torch_predictor.py', f"--model={self.known_args_dict['model']}", f"--checkpoint={checkpoint}",
                          f"--apply_sigmoid={self.unknown_args_dict['apply_sigmoid']}", f"--blob_threshold={self.unknown_args_dict['blob_threshold']}")
            else:
                # TF
                os.system('python', 'tf_predictor.py', f"--model={self.known_args_dict['model']}", f"--checkpoint={checkpoint}",
                          f"--apply_sigmoid={self.unknown_args_dict['apply_sigmoid']}")
            
            # create submission file
            experiment_submission = os.path.join(self.submission_folder, f"{exp_name}.csv")
            mask_to_submission.flags.DEFINE_string("submission_filename", experiment_submission, "The output csv for the submission.")
            os.system('python', 'mask_to_submission.py')    
        
        self.submission()
    
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
            
            trainer.train()
            f1_eval = trainer.eval()['f1_score']
            best_model_checkpoint = trainer.curr_best_checkpoint_path
            if best_model_checkpoint is None:
                self.experiment_name
                curr_run_idx += 1
                print("No checkpoint was saved. Omitting this run")
                continue
            self.checkpoint_paths.append(best_model_checkpoint)
            self.experiment_names.append(new_run_name)
            self.model_weights.append(f1_eval)
            
            # calculate daat errors
            data_inverse_f1_scores = self.trainer.weights
            # normalize between 0 and 1
            set_during_training = data_inverse_f1_scores[data_inverse_f1_scores != 2]
            normalized = set_during_training / (
                    2 * np.max(np.absolute(set_during_training)))  # between -0.5 and +0.5
            data_inverse_f1_scores[data_inverse_f1_scores != 2] = normalized + 0.5  # between 0 and 1
            # probability of zero is not wished for
            # values that have not been set in the training process just receive the old sampling probability
            data_inverse_f1_scores[data_inverse_f1_scores == 2] = self.dataloader.weights[self.weights == 2]
            weights_groundtruth_term = (1-data_inverse_f1_scores)*(-1) + data_inverse_f1_scores
            
            # calculate model error 
            model_error = np.sum(self.dataloader.weights*weights_groundtruth_term) / np.sum(self.dataloader.weights)
            total_model_performance = 0.5 * np.log((1-model_error)/model_error)
            self.model_weights.append(total_model_performance)
            
            # calculate new data weights
            self.dataloader.weights = self.dataloader.weight*np.exp(total_model_performance*weights_groundtruth_term)
            
            self.update_files() # save only if training went through
            curr_run_idx += 1
    
    
    def get_hyperparams(self):
        ...
    
    def update_files(self):
        pickle.dump(self.checkpoint_paths, open(os.path.join(self.adaboost_run_path, "checkpoints.pkl", 'wb')))
        pickle.dump(self.experiment_names, open(os.path.join(self.adaboost_run_path, "experiments.pkl", 'wb')))
        pickle.dump(self.model_weights, open(os.path.join(self.adaboost_run_path, "model_weights.pkl", 'wb')))
        with open(os.path.join(self.adaboost_run_path, "data_weights.ply"), 'rb') as file:
            np.save(os.path.join(self.adaboost_run_path, "data_weights.ply"), self.dataloader.weights)
    
    def submission(self):
        # weighted majority voting
        output_filename = os.path.join(self.adaboost_run_path, "final_submission_file.csv")
        # ensembles 
        if not os.path.isdir(self.submission_folder):
            print(f'Directory "{self.submission_folder}" (supposed to contain CSV files created in self.evaluate()) '
                'does not exist')
            return
    
        csv_filenames = [f"{name}.csv" for name in self.experiment_names]
        if len(csv_filenames) % 2 == 0:
            print('WARNING: found even number of submissions to ensemble; not recommended; will use value 1 to break ties')

        patch_order_list = []
        patches = {}
        for file_idx, filename in tqdm(enumerate(csv_filenames)):
            model_weight = self.model_weights[file_idx]
            with open(os.path.join(self.submission_folder, filename), 'r') as file:
                file_str = file.read()
                for line in map(lambda l: l.strip().lower(), file_str.splitlines()):
                    if line.startswith('id') or line.startswith('#') or not ',' in line:
                        continue
                    key, value = line.split(',')
                    patches[key] = model_weight*(patches.get(key, 0) + (-1 + 2 * int(value)))
                    if file_idx == 0:
                        patch_order_list.append(key)
        
        patches = {k: 1 if v > 0 else 0 for k, v in patches.items()}
        out_lines = ['id,prediction', *[f'{k},{patches.get(k, 0)}' for k in patch_order_list]]
        with open(output_filename, 'w') as file:
            file.write('\n'.join(out_lines))
        print(f'Ensembled {len(csv_filenames)} submissions into "{output_filename}"')