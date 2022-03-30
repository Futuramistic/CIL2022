from utils import *
from zipfile import ZipFile
import mlflow
import tempfile
import shutil
import time
from typing import Dict, Any


def log_visualizations(trainer, iteration_index):
    """
    Log the segmentations to MLFlow as images
    """
    eval_start = time.time()
    # store segmentations in temp_dir, then upload temp_dir to Mlflow server
    temp_dir = tempfile.mkdtemp()

    eval_inference_start = time.time()
    images = trainer.create_visualizations()
    trainer.save_image_array(images, temp_dir)
    eval_inference_end = time.time()

    # MLFlow does not have the functionality to log artifacts per training step,
    # so we have to incorporate the training step (iteration_idx) into the artifact path
    eval_mlflow_start = time.time()
    mlflow.log_artifacts(temp_dir, 'training/iteration_%07i' % iteration_index)
    eval_mlflow_end = time.time()

    shutil.rmtree(temp_dir)
    eval_end = time.time()

    if MLFLOW_PROFILING:
        print(f'\nEvaluation took {"%.4f" % (eval_end - eval_start)}s in total; '
              f'inference took {"%.4f" % (eval_inference_end - eval_inference_start)}s; '
              f'MLflow logging took {"%.4f" % (eval_mlflow_end - eval_mlflow_start)}s '
              f'(processed {trainer.num_samples_to_visualize} sample(s))')


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
    print('\nLogging codebase to MLFlow...')
    mlflow.log_artifact(CODEBASE_SNAPSHOT_ZIP_NAME, f'codebase/')
    print('Logging codebase successful')
    os.remove(CODEBASE_SNAPSHOT_ZIP_NAME)


def log_checkpoints():
    """
    Log a checkpoint to MLFlow
    """
    print('\nLogging checkpoints to MLFlow...')
    mlflow.log_artifact(CHECKPOINTS_DIR, '')
    print('Logging checkpoints successful')
    shutil.rmtree(CHECKPOINTS_DIR)  # Remove the directory and its contents
    os.makedirs(CHECKPOINTS_DIR)  # Recreate an empty directory


def log_metrics(metrics: Dict[str, Any]):
    """
    Log metrics to MLFlow
    Args:
        metrics (Dict[str, Any]): dictionary of metrics to log
    """
    mlflow.log_metrics(metrics)


def log_hyperparams(hyperparams: Dict[str, Any]):
    """
    Log hyperparameters to tensorflow

    Args:
        hyperparams (Dict[str, Any]): dictionary of hyperparameters to log
    """
    mlflow.log_params(hyperparams)
