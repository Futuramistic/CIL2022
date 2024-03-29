from trainers.trainer_torch import TorchTrainer
from utils.logging import pushbullet_logger
from utils import ROOT_DIR
from tqdm import tqdm

import os
import pickle
import numpy as np
import time
    
SMOL = 2e-5
DEFAULT_MONOBOOST_TEMPERATURE = 0.0001

class AdaBooster:
    def __init__(self, factory, known_args_dict, unknown_args_dict,  model_spec_args, trainer_spec_args,
                 dataloader_spec_args, monoboost, monoboost_temperature, is_debug):
        """An Adaboost class for performing the classical adaboost algorithm and other variants, such as the MonoBoost. The evaluation parameter will create the submission files on
        the test data set.
        Note that it is necessary to call each new adaboost experiment by a separate experiment name, otherwise the previous experiment
        will be loaded and continued

        Args:
            arguments are passed through the main function in main.py and were created from the user's inputs
            for evaluation, optionally use the argument '--apply_sigmoid' (bool), and if torch is used, additionally '--blob_threshold' (int)
        """
        self.factory = factory
        self.known_args_dict = known_args_dict
        self.unknown_args_dict = unknown_args_dict
        self.model_args = model_spec_args
        self.trainer_args = trainer_spec_args
        self.dataloader_args = dataloader_spec_args
        self.monoboost = monoboost
        self.monoboost_temperature = monoboost_temperature
        self.is_debug = is_debug
        
        # Create the dataloader using the commandline arguments
        self.dataloader = factory.get_dataloader_class()(**dataloader_spec_args)
        self.test_dataloader = factory.get_dataloader_class()(dataset="original")
        
        # load last adaboost setting if adaboost name already exists, otherwise create new file structure for current experiment
        self.checkpoint_paths = [] # checkpoint paths for best epoch of trained models
        self.experiment_names = [] # experiment names for trained models
        self.model_weights = [] # model weight (unnormalized) depending on the models overall performance
        # sample weights are managed by the dataloader in order to be able to return the trainer the needed data structures
        self.adaboost_run_path = os.path.join(ROOT_DIR, "adaboost/adaboost_runs", self.known_args_dict['experiment_name'])
        self.submission_folder = os.path.join(self.adaboost_run_path, "submission")
        self.submission_folder_from_root = os.path.join("adaboost/adaboost_runs", self.known_args_dict['experiment_name'], "submission")
        if os.path.isdir(self.adaboost_run_path):
            self.checkpoint_paths = pickle.load(open(os.path.join(self.adaboost_run_path, 'checkpoints.pkl'), 'rb'))
            self.experiment_names = pickle.load(open(os.path.join(self.adaboost_run_path, 'experiments.pkl'), 'rb'))
            self.model_weights = pickle.load(open(os.path.join(self.adaboost_run_path, 'model_weights.pkl'), 'rb'))
            with open(os.path.join(self.adaboost_run_path, "data_weights.ply"), 'rb') as file:
                # data weights don't vary in size and are easier to work with as np arrays 
                self.dataloader.weights = np.load(file, allow_pickle=True).astype(float)
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
            
            pushbullet_logger.send_pushbullet_message('Training finished.' + \
                                                        f'Hyperparameters:\n{self.get_hyperparams()}')
    
    def train(self):
        # training loop
        curr_run_idx = len(self.experiment_names)
        if curr_run_idx > 0:
            self.dataloader.weights_set = True

        if self.monoboost:
            # calculate the val sample -- train sample similarities once

            dataset_name = self.dataloader.dataset
            val_sample_train_sample_dist_path =\
                os.path.join('dataset', dataset_name, f'precached_sample_distances__{dataset_name}__{dataset_name}.pkl')

            # Create the model using the commandline arguments
            model = self.factory.get_model_class()(**self.model_args)

            # Create the trainer using the commandline arguments
            trainer_class = self.factory.get_trainer_class()
            trainer = trainer_class(dataloader=self.dataloader, model=model,
                                    **self.trainer_args)

            if os.path.isfile(val_sample_train_sample_dist_path):
                with open(val_sample_train_sample_dist_path, 'rb') as f:
                    training_sample_weights_for_validation_samples = pickle.load(f)
            else:
                print('Precaching val sample -- train sample distances...')

                # iterate through, calculate
                sample_dist_path = os.path.join('dataset', dataset_name,
                                                f'sample_distances__{dataset_name}__{dataset_name}.pkl')

                if not os.path.isfile(sample_dist_path):
                    print(f'Sample distance file ({sample_dist_path}) not found; creating file. This could take some time.')
                    os.system(f'python -m processing.sample_feature_dist_calculator'
                            f' --dataset_1={dataset_name} --dataset_2={dataset_name}')
                
                with open(sample_dist_path, 'rb') as f:
                    sample_distances = pickle.load(f)
                
                train_dl = self.dataloader.get_training_dataloader(trainer.split, batch_size=1, weights=None,
                                                                   preprocessing=trainer.preprocessing,
                                                                   suppress_adaboost_weighting=True, shuffle=False)
                test_dl = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                preprocessing=trainer.preprocessing)

                training_sample_weights_for_validation_samples = np.zeros((len(test_dl), len(train_dl)))


                for test_idx in range(len(test_dl)):
                    # test_idx = test_idx_with_offset - len(train_dl)
                    test_x_filename = os.path.basename(self.dataloader.training_img_paths[len(train_dl) + test_idx])
                    for train_idx in range(len(train_dl)):
                        train_x_filename = os.path.basename(self.dataloader.training_img_paths[train_idx])
                        training_sample_weights_for_validation_samples[test_idx][train_idx] =\
                            sample_distances[train_x_filename][test_x_filename]
                    
                    training_sample_weights_for_validation_samples[test_idx] = (
                        np.exp(-training_sample_weights_for_validation_samples[test_idx] / self.monoboost_temperature) /
                        np.sum(np.exp(-training_sample_weights_for_validation_samples[test_idx]))
                               / self.monoboost_temperature)
                
                print('Val sample -- train sample distances calculated')
                with open(val_sample_train_sample_dist_path, 'wb') as f:
                    pickle.dump(training_sample_weights_for_validation_samples, f)
                    
        global CHECKPOINTS_DIR
        while curr_run_idx < self.known_args_dict["adaboost_runs"]:
             # snapshot zip name and session ID don't have to be set, as code doesn't change
            CHECKPOINTS_DIR = os.path.join("checkpoints", str(int(time.time() * 1000)))
            
            if not self.monoboost:
                # Create the model using the commandline arguments
                model = self.factory.get_model_class()(**self.model_args)
                
                # Create the trainer using the commandline arguments
                trainer_class = self.factory.get_trainer_class()
                trainer = trainer_class(dataloader=self.dataloader, model=model,
                                        **self.trainer_args)
            
            # simply keep using old model if we're using MonoBoost

            trainer.best_f1_score = -1.0

            # adapt the mlflow experiment name for each new model
            old_run_name = trainer.mlflow_experiment_name
            new_run_name = ("AdaBoost_" if not self.monoboost else "MonoBoost_") + f"{str(old_run_name)}_Run_{curr_run_idx}" if old_run_name is not None else f"Run_{curr_run_idx}"
            trainer.mlflow_experiment_name = new_run_name
            
            trainer.train()
            f1_eval = trainer.eval()['f1_weighted_scores']  #['f1_road_scores']
            best_model_checkpoint = trainer.curr_best_checkpoint_path
            if best_model_checkpoint is None:
                curr_run_idx += 1
                print("No checkpoint was saved. Omitting this run")
                continue
            self.checkpoint_paths.append(best_model_checkpoint)
            self.experiment_names.append(f"{trainer.mlflow_experiment_name}_{curr_run_idx}")
            self.model_weights.append(f1_eval)
            
                
            if self.monoboost:
                dataset_name = self.dataloader.dataset
                
                # loop through all train samples for one epoch; shuffling will not affect us as long as we know the sample indices
                train_dl = self.dataloader.get_training_dataloader(trainer.split, trainer.batch_size, weights=None, preprocessing=trainer.preprocessing, suppress_adaboost_weighting=True)
                
                f1_weighted_scores = trainer.get_F1_scores_validation()
                val_f1_errors = 1.0 - np.asarray(f1_weighted_scores, dtype=float)
                
                transp = np.transpose(training_sample_weights_for_validation_samples)
                self.dataloader.weights = transp @ val_f1_errors
                
                # calculate model error (between 0 and 1)
                model_error = np.mean(val_f1_errors)
                log_term = (1 - model_error) / model_error
                if log_term <= 0.0:
                    # infinity numerically unstable
                    log_term = np.float64(SMOL)
                total_model_performance = 0.5 * np.log(log_term)  # the higher, the better
                self.model_weights.append(total_model_performance)
            else:
                data_f1_scores = np.asarray(trainer.get_F1_scores_training_no_shuffle(), dtype=float)
                
                # values that have not been set in the training process just receive the old sampling probability
                print(data_f1_scores)
                print(self.dataloader.weights)
                # data_f1_scores[data_f1_scores == 2] = self.dataloader.weights[data_f1_scores == 2]
                
                # the error of the samples (1-F1 of each sample)
                # old:
                # samples_error_term = (-data_f1_scores) + (1-data_f1_scores)

                # instead of simulating the classical classifier-based AdaBoost (-1 for incorrect, 1 for correct),
                # we renormalize each weight to be between 0 and 1 so that model_error stays between 0 and 1
                samples_error_term = 1 - data_f1_scores
                
                # calculate model error (between 0 and 1)
                model_error = np.sum(self.dataloader.weights * samples_error_term) / np.sum(self.dataloader.weights)
                log_term = (1 - model_error) / model_error
                if log_term <= 0.0:
                    # infinity numerically unstable
                    log_term = np.float64(SMOL)
                total_model_performance = 0.5 * np.log(log_term)  # the higher, the better
                self.model_weights.append(total_model_performance)
                
                # calculate new data weights
                # weight samples based on error and performance of this model
                self.dataloader.weights = self.dataloader.weights*np.exp(total_model_performance * samples_error_term)
                # normalize between 0 and 1 again (no negative values allowed)
                # normalized = self.dataloader.weights / (
                #         2 * np.max(np.absolute(self.dataloader.weights)))  # between -0.5 and +0.5
                # self.dataloader.weights = normalized + 0.5  # between 0 and 1
                
            self.dataloader.weights[self.dataloader.weights <= 0] = SMOL  # avoid 0
            
            print('zip(self.dataloader.training_gt_paths, self.dataloader.weights): ',
                  list(zip(self.dataloader.training_img_paths, self.dataloader.weights)))
            
            # normalize so sum of weights is 1
            #self.dataloader.weights /= sum(self.dataloader.weights)
           
            # save
            self.update_files()
            
            if curr_run_idx == 0:
                self.dataloader.weights_set = True
            curr_run_idx += 1
    
    def evaluate(self):
        # creates single submission and calls ensembled submission
        self.return_codes = []
        for idx, checkpoint in enumerate(self.checkpoint_paths):
            exp_name = self.experiment_names[idx]
            trainer_class = self.factory.get_trainer_class()
            
            # create predictions on test data set
            if issubclass(trainer_class, TorchTrainer):
                # Torch
                command = f"python torch_predictor.py --model={self.known_args_dict['model']} --checkpoint={checkpoint}"
                if 'apply_sigmoid' in self.known_args_dict.keys():
                    command += " --apply_sigmoid"
                if 'blob_threshold' in self.known_args_dict.keys():
                    command += f" --blob_threshold={self.unknown_args_dict['blob_threshold']}"
                return_code = os.system(command)
            else:
                # TF
                command = f"python tf_predictor.py --model={self.known_args_dict['model']} --checkpoint={checkpoint}"
                if 'apply_sigmoid' in self.known_args_dict.keys():
                    command += " --apply_sigmoid"
                return_code = os.system(command)
            self.return_codes.append(return_code)
            if return_code != 0:
                print("There has been an error in the tf/torch_predictor. This model will not be used for the ensembling")
                # create submission file
                continue
            os.makedirs(self.submission_folder, exist_ok=True)
            experiment_submission = os.path.join(self.submission_folder_from_root, f"{exp_name}.csv")
            print(experiment_submission)
            return_code = os.system(f"python mask_to_submission.py --submission_filename={experiment_submission}")
            if return_code != 0:
                print("There has been an error in the mask_to_submission.py. This model will not be used for the ensembling")
                self.return_codes[-1] = return_code
        
        self.submission()
    
    
    def get_hyperparams(self):
        ...
    
    def update_files(self):
        pickle.dump(self.checkpoint_paths, open(os.path.join(self.adaboost_run_path, "checkpoints.pkl"), 'wb'))
        pickle.dump(self.experiment_names, open(os.path.join(self.adaboost_run_path, "experiments.pkl"), 'wb'))
        pickle.dump(self.model_weights, open(os.path.join(self.adaboost_run_path, "model_weights.pkl"), 'wb'))
        with open(os.path.join(self.adaboost_run_path, "data_weights.ply"), 'wb') as file:
            np.save(file, self.dataloader.weights)
    
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
            if self.return_codes[file_idx] != 0:
                continue
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
