from datetime import datetime
import pysftp
import requests
from requests.auth import HTTPBasicAuth
import shutil
from urllib.parse import urlparse
from utils import *
import zipfile


"""
Script to download a scraped dataset from MLFlow
"""


curated = True  # set to True to download only curated part of dataset

mlflow_ftp_pass = requests.get(MLFLOW_FTP_PASS_URL, auth=HTTPBasicAuth(MLFLOW_HTTP_USER, MLFLOW_HTTP_PASS)).text

print(f'Requesting dataset generation (could take some time)...')

sftp_zip_path = requests.get(f'https://algvrithm.com/zip_scraped_dataset.php{"?curated=1" if curated else ""}',
                             timeout=600).text

print(f'Downloading dataset (curated: {curated}) from "{sftp_zip_path}"...')

url_components = urlparse(sftp_zip_path)
url = f'sftp://mlflow_user@algvrithm.com/mlruns/maps{"_curated" if curated else ""}'
local_dir = os.path.join('dataset', os.path.basename(url_components.path).replace('.zip', ''))
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None
zip_path = os.path.join(local_dir, 'dataset.zip')

if not os.path.isdir(local_dir):
    os.makedirs(local_dir, exist_ok=True)

with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_FTP_USER, password=mlflow_ftp_pass, cnopts=cnopts) as sftp:
    sftp.get(url_components.path, zip_path)

print('Extracting dataset...')

with zipfile.ZipFile(zip_path) as z:
    z.extractall(local_dir)

print('Removing ZIP file...')
os.unlink(zip_path)

if not os.path.isdir(os.path.join(local_dir, 'training')):
    os.makedirs(os.path.join(local_dir, 'training'), exist_ok=True)

if os.path.isdir(os.path.join(local_dir, 'images')):
    shutil.move(os.path.join(local_dir, 'images'), os.path.join(local_dir, 'training', 'images'))
    
if os.path.isdir(os.path.join(local_dir, 'groundtruth')):
    shutil.move(os.path.join(local_dir, 'groundtruth'), os.path.join(local_dir, 'training', 'groundtruth'))

print('Ensuring images and GT match...')

num_samples = 0
for folder_name, _, file_names in os.walk(os.path.join(local_dir, 'training', 'images')):
    for file_name in file_names:
        if file_name.endswith('.png'):
            if not os.path.isfile(os.path.join(local_dir, 'training', 'groundtruth', file_name)):
                os.unlink(os.path.join(local_dir, 'training', 'images', file_name))
            else:
                num_samples += 1

for folder_name, _, file_names in os.walk(os.path.join(local_dir, 'training', 'groundtruth')):
    for file_name in file_names:
        if file_name.endswith('.png'):
            if not os.path.isfile(os.path.join(local_dir, 'training', 'images', file_name)):
                os.unlink(os.path.join(local_dir, 'training', 'groundtruth', file_name))

ts_path = os.path.join(local_dir, "download_timestamp.txt")
with open(ts_path, "w") as file:
    file.write(str(datetime.now()))

print(f'Dataset saved to "{local_dir}"; number of samples: {num_samples}, of which {num_samples-25} training '
      f'({"%.5f" % ((num_samples-25)/num_samples)})')
