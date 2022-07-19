import argparse
import hashlib
import os
import sys
import pysftp
import requests

from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
from factory import *
from utils import *


"""
Given a list of sftp urls, retrieve the corresponding checkpoints, load them into their respective models
and let each model make a prediction, before ensembling all predictions into a single submission file
"""


os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_HTTP_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_HTTP_PASS


def sftp_url_to_command(url):
    """
    Transform an sftp url into an sftp command
    Args:
        url (str): the url to transform
    """
    parts = url.split(':')
    command = f'sftp mlflow_user@algvrithm.com:/{parts[3][3:]}'
    return command


def load_checkpoint(checkpoint_path, checkpoints_output_dir):
    """
    Download a model checkpoint either locally or from the network
    Args:
        checkpoint_path (str): The checkpoint url
    """
    load_from_sftp = checkpoint_path.lower().startswith('sftp://')
    if load_from_sftp:
        original_checkpoint_hash = hashlib.md5(str.encode(checkpoint_path)).hexdigest()
        extension = 'pt' if checkpoint_path.lower().endswith('.pt') else 'ckpt'
        final_checkpoint_path = f'{checkpoints_output_dir}/checkpoint_{original_checkpoint_hash}.{extension}'
        condition = os.path.isfile(final_checkpoint_path) if extension == 'pt' else os.path.isdir(final_checkpoint_path)
        if not condition:
            if extension != 'pt':
                os.makedirs(final_checkpoint_path, exist_ok=True)
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL,
                                           auth=HTTPBasicAuth(os.environ['MLFLOW_TRACKING_USERNAME'],
                                                              os.environ['MLFLOW_TRACKING_PASSWORD'])).text
            url_components = urlparse(checkpoint_path)
            with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_FTP_USER, password=mlflow_ftp_pass,
                                   cnopts=cnopts) as sftp:
                callback = lambda x, y: print('Downloading checkpoint %s [%d%%]\r' %
                                              (original_checkpoint_hash, int(100 * float(x) / float(y))), end="")
                if extension == 'pt':
                    sftp.get(url_components.path, final_checkpoint_path, callback=callback)
                else:
                    sftp_download_dir_portable(sftp, remote_dir=url_components.path, local_dir=final_checkpoint_path,
                                               callback=callback)
            print(f'\nDownload successful')
        else:
            print(f'Checkpoint "{checkpoint_path}", to be downloaded to "{final_checkpoint_path}", found on disk\n')
        return final_checkpoint_path


def main(model_name, sftp_paths_file, checkpoints_output_dir, output_predictions_dir):
    if model_name is not None:
        model_name = model_name.lower().replace('-', '').replace('_', '')

    # Make sure the user wants to overwrite the current checkponts directory
    if os.path.exists(checkpoints_output_dir):
        answer = input(f'{checkpoints_output_dir} already exists. Do you want to overwrite it? [y/n]')
        if answer.lower() == 'y':
            shutil.rmtree(checkpoints_output_dir)
        else:
            print('Exiting...')
            return
    os.mkdir(checkpoints_output_dir)

    # Read the checkpoint paths
    current_model_name = options.model
    sftp_paths = []
    sftp_path_model_names = []
    with open(sftp_paths_file) as file:
        for line in map(lambda l: l.strip(), file.readlines()):
            if '#' in line: 
                line = line[:line.index('#')].strip()
            if line == '':
                continue
            if line.endswith(':'):
                current_model_name = line.replace(':', '')
            else:
                sftp_paths.append(line)
                sftp_path_model_names.append(current_model_name)

    # Load the checkpoints from paths
    checkpoint_paths = []
    for sftp_path in sftp_paths:
        checkpoint_paths.append(load_checkpoint(sftp_path, checkpoints_output_dir))

    if os.path.exists(output_predictions_dir):
        shutil.rmtree(output_predictions_dir)
    os.mkdir(output_predictions_dir)

    # Make one prediction per checkpoint
    for idx, checkpoint in enumerate(checkpoint_paths):
        fact = Factory.get_factory(model_name=sftp_path_model_names[idx])
        # also captures torch RL models with different trainer:
        is_torch = issubclass(fact.get_dataloader_class(), TorchDataLoader)
        predictor_script = 'torch_predictor' if is_torch else 'tf_predictor'
        command = f"python {predictor_script}.py -m {model_name} -c {checkpoints_output_dir}/{checkpoint}"
        if model_name in ['deeplabv3', 'cranet']:
            command += ' --apply_sigmoid'
        os.system(command)
        os.system('python mask_to_submission.py')
        move_command = 'move' if os.name == 'nt' else 'mv'
        os.system(f'{move_command} dummy_submission.csv {output_predictions_dir}/{idx}.csv')

    # Ensemble all predictions
    os.system(f'python -m processing.ensembled_submission_creator')


if __name__ == '__main__':
    desc_str = 'Given a list of sftp urls, retrieve the corresponding checkpoints, load them into their respective' \
               ' models and let each model make a prediction, before ensembling all predictions into a single ' \
               ' submission file'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-m', '--model', type=str, default=None, help='Default model name (can specify different '
                                                                      'names in sftp_paths_file)')
    parser.add_argument('-f', '--sftp_paths_file', required=False, default='sftp_paths.txt', type=str,
                        help='Path to file with SFTP paths of checkpoints to use to create ensembles')
    parser.add_argument('-c', '--checkpoints_output_dir', required=False, default='model_checkpoints', type=str,
                        help='Path to directory in which to store downloaded model checkpoints')
    parser.add_argument('-o', '--output_predictions_dir', required=False, default='submissions_to_ensemble', type=str,
                        help='Path to directory in which to store submissions to ensemble')

    options = parser.parse_args()
    main(options.model, options.sftp_paths_file, options.checkpoints_output_dir, options.output_predictions_dir)
