import os
import sys
import time
import argparse
import shutil
import pysftp
import hashlib
import requests
from requests.auth import HTTPBasicAuth
from utils import *
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_HTTP_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_HTTP_PASS

sftp_paths_file = 'sftp_paths.txt'
checkpoints_output_dir = 'model_checkpoints'
output_predictions_dir = 'submissions_to_ensemble'


def sftp_url_to_command(url):
    parts = url.split(':')
    command = f'sftp mlflow_user@algvrithm.com:/{parts[3][3:]}'
    return command


def load_checkpoint(checkpoint_path, model_type):
    load_from_sftp = checkpoint_path.lower().startswith('sftp://')
    if load_from_sftp:
        original_checkpoint_hash = hashlib.md5(str.encode(checkpoint_path)).hexdigest()
        extension = 'pt' if model_type == 'torch' else 'ckpt'
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


def main():
    parser = argparse.ArgumentParser(description='Downloads the given checkpoints and create an ensembled prediction')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-t', '--model_type', type=str, required=True, help='Model type in [torch, tensorflow]')
    options = parser.parse_args()
    model_name = options.model
    model_type = options.model_type
    model_name = model_name.lower().replace('-', '').replace('_', '')

    # if os.path.exists(checkpoints_output_dir):
    #     answer = input(f'{checkpoints_output_dir} already exists. Do you want to overwrite it? [y/n]')
    #     if answer.lower() == 'y':
    #         shutil.rmtree(checkpoints_output_dir)
    #     else:
    #         print('Exiting...')
    #         exit(-1)
    # os.mkdir(checkpoints_output_dir)
    #
    # with open(sftp_paths_file) as file:
    #     sftp_paths = file.readlines()
    #
    # for sftp_path in sftp_paths:
    #     load_checkpoint(sftp_path, model_type=model_type)

    if os.path.exists(output_predictions_dir):
        shutil.rmtree(output_predictions_dir)
    os.mkdir(output_predictions_dir)

    predictor_script = 'torch_predictor' if model_type == 'torch' else 'tf_predictor'
    for i, checkpoint in enumerate(os.listdir(checkpoints_output_dir)):
        command = f"python {predictor_script}.py -m {model_name} -c {checkpoints_output_dir}/{checkpoint}"
        if model_name == 'deeplabv3':
            command += ' --apply_sigmoid'
        os.system(command)
        os.system('python mask_to_submission.py')
        move_command = 'move' if os.name == 'nt' else 'mv'
        os.system(f'{move_command} dummy_submission.csv {output_predictions_dir}/{i}.csv')

    os.system(f'python ensembled_submission_creator.py')


if __name__ == '__main__':
    main()
