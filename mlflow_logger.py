from utils import *
from zipfile import ZipFile
import mlflow
import tempfile
import shutil
import time


def log_visualizations(callback):
    """
    Log the segmentations to MLFlow as images
    """
    eval_start = time.time()
    # store segmentations in temp_dir, then upload temp_dir to Mlflow server
    temp_dir = tempfile.mkdtemp()

    eval_inference_start = time.time()
    callback.create_visualizations(temp_dir)
    eval_inference_end = time.time()

    # MLflow does not have the functionality to log artifacts per training step,
    # so we have to incorporate the training step (iteration_idx) into the artifact path
    eval_mlflow_start = time.time()
    mlflow.log_artifacts(temp_dir, 'training/iteration_%07i' % callback.iteration_idx)
    eval_mlflow_end = time.time()

    shutil.rmtree(temp_dir)
    eval_end = time.time()

    if MLFLOW_PROFILING:
        print(f'\nEvaluation took {"%.4f" % (eval_end - eval_start)}s in total; '
              f'inference took {"%.4f" % (eval_inference_end - eval_inference_start)}s; '
              f'MLflow logging took {"%.4f" % (eval_mlflow_end - eval_mlflow_start)}s '
              f'(processed {callback.trainer.num_samples_to_visualize} sample(s))')


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
    mlflow.log_artifact(CODEBASE_SNAPSHOT_ZIP_NAME, f'codebase/')
    os.remove(CODEBASE_SNAPSHOT_ZIP_NAME)
