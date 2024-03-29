"""This file handles the logging to mlflow (all checkpoints, code base, command, logs and metrics"""
import mlflow
import tempfile
import shutil
import shlex

from utils import *
from zipfile import ZipFile
from typing import Dict, Any


def logging_to_mlflow_enabled():
    """
    Returns whether or not it is possible to log to MLFlow
    """
    return mlflow.active_run() is not None


def log_visualizations(trainer, iteration_index, epoch_idx, epoch_iteration_idx):
    """
    Log the segmentations to MLFlow as images.
    Called by a Trainer, and calls that Trainer's "create_visualizations" (which contains ML framework-specific code).
    See the documentation of "create_visualizations" for more information.
    Args:
        trainer (Trainer): The trainer we are using
        iteration_index (int): The current iteration (global)
        epoch_idx (int): The current epoch
        epoch_iteration_idx (int): The current iteration in the current epoch
    """
    if not logging_to_mlflow_enabled():
        return False

    eval_start = time.time()
    # store segmentations in temp_dir, then upload temp_dir to Mlflow server
    temp_dir = tempfile.mkdtemp()

    eval_inference_start = time.time()

    # MLFlow does not have the functionality to log artifacts per training step,
    # so we have to incorporate the training step (iteration_idx) into the filename
    vis_file_path = os.path.join(temp_dir, 'iteration_%07i.png' % iteration_index)
    vis_file_path = next(filter(lambda x: x is not None,
                                [trainer.create_visualizations(vis_file_path, iteration_index, epoch_idx,
                                                               epoch_iteration_idx), vis_file_path]))
    eval_inference_end = time.time()

    eval_mlflow_start = time.time()
    # log the file to the run's root directory for quicker access
    mlflow.log_artifact(vis_file_path, '')
    eval_mlflow_end = time.time()

    shutil.rmtree(temp_dir)
    eval_end = time.time()

    if MLFLOW_PROFILING:
        print(f'\nEvaluation took {"%.4f" % (eval_end - eval_start)}s in total; '
              f'inference took {"%.4f" % (eval_inference_end - eval_inference_start)}s; '
              f'MLflow logging took {"%.4f" % (eval_mlflow_end - eval_mlflow_start)}s '
              f'(processed {trainer.num_samples_to_visualize} sample(s))')

    return True


def snapshot_codebase():
    """
    Zip codebase python files
    """
    with ZipFile(CODEBASE_SNAPSHOT_ZIP_NAME, 'w') as zip_obj:
        for folder_name, _, file_names in os.walk('.'):
            for file_name in file_names:
                if file_name.endswith('.py'):
                    file_path = os.path.join(folder_name, file_name)
                    zip_obj.write(file_path)


def log_codebase():
    """
    Log the codebase to MLFlow
    """
    if logging_to_mlflow_enabled():
        print('\nLogging codebase to MLFlow...')
        mlflow.log_artifact(CODEBASE_SNAPSHOT_ZIP_NAME, f'codebase/')
        print('Logging codebase successful')
    else:
        print('Cannot log codebase to MLFlow, as logging is disabled')

    os.remove(CODEBASE_SNAPSHOT_ZIP_NAME)


def log_checkpoints(remove_local=True):
    """
    Log a checkpoint to MLFlow
    Args:
        remove_local (bool, optional): Whether to remove the local checpoint after successfully logging to mlflow.
                                       Defaults to True
    """
    if logging_to_mlflow_enabled():
        # check if there are checkpoints to log
        if os.path.isdir(CHECKPOINTS_DIR) and len(os.listdir(CHECKPOINTS_DIR)) > 0:
            print('\nLogging checkpoints to MLFlow...')
            
            mlflow.log_artifacts(CHECKPOINTS_DIR, 'checkpoints/')
            
            print('Logging checkpoints successful')
            if remove_local:
                shutil.rmtree(CHECKPOINTS_DIR)  # Remove the directory and its contents
                os.makedirs(CHECKPOINTS_DIR)  # Recreate an empty directory
    else:
        print('Cannot log checkpoint to MLFlow, as logging is disabled')


def log_artifact(path, artifact_dir='', emit_warning_if_logging_disabled=True):
    """
    Log an artifact to MLFLow
    Args:
        path (Path or String): path to file/directory to log
        artifact_dir (Path or String): remote run subdirectory to log the file/directory into ('' for root run directory)
        emit_warning_if_logging_disabled (bool, optional): True if a warning should be emitted in case logging to MLFlow 
                                          is disabled, and False otherwise. Defaults to True
    """
    if logging_to_mlflow_enabled():
        mlflow.log_artifact(path, artifact_dir)
    elif emit_warning_if_logging_disabled:
        print(f'Cannot log "{path}" to MLFlow, as logging is disabled')


def log_metrics(metrics: Dict[str, Any], aggregate_iteration_idx: int):
    """
    Log metrics to MLFlow
    Args:
        metrics (Dict[str, Any]): dictionary of metrics to log
        aggregate_iteration_idx (int): index of current aggregate training iteration
    """

    # do not emit a warning here, as this is likely to spam the console
    if logging_to_mlflow_enabled():
        mlflow.log_metrics(metrics, step=aggregate_iteration_idx)


def log_hyperparams(hyperparams: Dict[str, Any]):
    """
    Log hyperparameters to MLFlow

    Args:
        hyperparams (Dict[str, Any]): dictionary of hyperparameters to log
    """

    if logging_to_mlflow_enabled():
        mlflow.log_params(hyperparams)
    else:
        print('Cannot log checkpoint to MLFlow, as logging is disabled')


def log_logfiles():
    """
    Log this session's "stderr.log" and "stdout.log" to MLflow
    """
    # do not emit a warning here, as this is likely to spam the console
    if logging_to_mlflow_enabled():
        stderr_path = os.path.join(ROOT_DIR, LOGGING_DIR, f'stderr_{SESSION_ID}.log')
        stdout_path = os.path.join(ROOT_DIR, LOGGING_DIR, f'stdout_{SESSION_ID}.log')
        for path in [stderr_path, stdout_path]:
            if os.path.isfile(path):
                mlflow.log_artifact(path, 'logs/')


def log_command_line():
    """
    Log the command line that was executed to run the script
    """
    if logging_to_mlflow_enabled():
        cmdline = " ".join(map(shlex.quote, sys.argv[1:]))
        mlflow.log_text(cmdline, COMMAND_LINE_FILE_NAME)
