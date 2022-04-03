# ☁️ Working with Euler

## 🌐 Connecting to Euler

- on Linux machines: use native ```ssh``` command
- on Windows machines: recommend using WSL (Windows Subsystem for Linux; https://docs.microsoft.com/en-us/windows/wsl/install) and ```ssh```, or, alternatively, using PuTTY (https://www.putty.org/)
- make sure you're connected to the ETHZ VPN using a VPN adapter, else the SSH connection will fail
  - go to https://sslvpn.ethz.ch/
  - log in using \<username\>@student-net.ethz.ch and your ETHZ network password
  - click "Download for Windows" / "Download for Linux"
  - install, start and connect
  - on Windows, the VPN client tends to be unstable; you may have to disconnect and reconnect at times if some network operation seems to be stuck
- connect to a specific login node: ```ssh <ethz-username>@<login-node>.euler.ethz.ch```
- connect to a randomly chosen node: ```ssh <ethz-username>@euler.ethz.ch```
- password: standard (non-network) ETHZ password
- to show name of current node, run ```hostname```
- to disconnect, run ```exit```

## 🛠️ Setting up working environment

- the setup only has to be done once, but parts described here (e.g. activating the Python virtual environment) are referenced later
- cloning the repository to your Euler storage
  - create GitHub personal access token: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
  - ```git clone https://github.com/Futuramistic/CIL2022.git```
    - username: GitHub username
    - password: personal access token
  - your files are not limited to a specific login node; they can be accessed from any node
- creating Python virtual environment
  - Conda is not available on Euler, so we have to use native Python venvs
  - activate Python 3.9.9: ```module load gcc/8.2.0 && module load python_gpu/3.9.9```
    - we need to use 3.9.9 due to incompatibilities of earlier Python versions installed on the cluster with some parts of our codebase 
  - ```cd``` to desired parent directory of environment (preferably ```~/```)
  - ```python -m venv <venv-name>```
    - in the following, we will use ```python -m venv venv_CIL2022```
- activating Python virtual environment
  - activate Python 3.9.9: ```module load gcc/8.2.0 && module load python_gpu/3.9.9```
  - run ```source <venv-dir>/bin/activate```
    - e.g. ```source ~/venv_CIL2022/bin/activate```
  - terminal should now have venv name as prefix
- installing requirements
  - make sure correct venv is installed and activated
  - ```cd``` into the ```CIL2022``` directory (where ```requirements.txt``` is located)
  - run ```pip install -r requirements.txt```
    - there may be errors related to ```pywin32``` because we're on a Linux machine
      - run ```nano requirements.txt```
      - move caret to ```wexpect``` line; remove or comment it
      - ```CTRL+O, Enter``` to save file
      - ```CTRL+X``` to exit nano
- generating jumpbox SSH key for MLflow
  - background:
    - Euler compute nodes (where jobs are executed) cannot connect to the internet by default
    - an HTTP proxy is available, yet other services (e.g. SSH) do not work without further adjustments
    - since internet access is available on the login nodes, the compute nodes will use them as a proxy ("jumpbox") when establishing SFTP connections to the MLflow server to upload artifacts
    - we need to generate SSH key pairs so that the compute node can trust the login node and vice versa
  - pick one login node (recommend ```eu-login-01```) and ```ssh``` into it
    - warning: this will have to be the login node referenced by ```MLFLOW_JUMP_HOST``` in ```utils.py```; currently, this is ```eu-login-01```, so it's strongly recommended to use that one
    - see "connect to a specific login node" under "Connecting to Euler"
  - run ```ssh-keygen -t ed25519 -f ~/.ssh/id_$(hostname)```
    - run this command as-is; the hostname will be determined automatically
  - do not enter a passphrase when prompted
  - run ```cat ~/.ssh/id_$(hostname).pub >> ~/.ssh/authorized_keys``` to authorize the generated public key

## ▶️ Starting a training run

- connect to Euler, as described above (not necessarily any specific node)
- activate the Python virtual environment, as described above
  - do not forget to load Python 3.9.9 before activating the venv!
- activate internet on compute nodes: ```module load eth_proxy```
- ```cd``` into ```CIL2022``` directory
- submit job using ```bsub```
  - for details and a list of available GPUs, see https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs
  - in general, the better the GPU, the longer the job will stay queued
  - do not forget to add ```NVIDIA``` before the GPU name, else the job will stay queued forever!
  - example: ```bsub -n 1 -W 12:00 -R "rusage[ngpus_excl_p=1, mem=4096]" -R "select[gpu_model0==NVIDIAGeForceGTX1080]" "python main.py --model=unet --n_channels=3 --n_classes=2"```
    - find node with at least 4096MB of CPU RAM and a GTX 1080; run ```python main.py --model=unet --n_channels=3 --n_classes=2```; limit runtime to 12 hours
- to show list of pending/running jobs with current status, run ```bjobs```
  - ```PEND``` means your job is still queued, ```RUN``` means it's running
  - after completion/failure, the job will be removed from the job list
- if everything went well, within a few minutes after the job is dequeued and starts running, you will be able to see and track the run in MLflow
  - before, dataset needs to be downloaded to the compute node, which takes time
- to kill a job, run ```bkill <job-id>```
- once finished or failed, the job will produce a log file (based on its name) in the current directory, containing the output of the job
  - to see all files in current directory, run ```ll```
  - the file should be called ```lsf.o<job-id>```
  - to inspect the file, run ```more lsf.o<job-id>```

## 🔄 Easy file synchronization with FileZilla
- working on files using SSH and the terminal can be tedious
  - alternative would be to ```git push``` and ```git pull``` on the login node everytime, but also tedious and spams commit history
- instead, we can work on files locally, and use SFTP to copy them to/from the login nodes
- download, install and open FileZilla client (available for Linux and Windows): https://filezilla-project.org/
- add a new server using ```File -> Site Manager -> New Site```; call it e.g. "Euler"
- as protocol, select SFTP
- as host, enter ```euler.ethz.ch```
- as port, enter ```22```
- as logon type, select "Normal"
- enter your ETHZ username and password
- click "OK"
- to connect, click the arrow next to the site manager button (leftmost button in icon menu) and select the site you just added (e.g. "Euler")
  - remember that you need to be connected to the ETHZ VPN to be able to connect to Euler
- when prompted whether to trust the host key, check "Always trust this host" and click "OK"
- you should now be able to browse your local filesystem in the left side of the window, and the remote filesystem in the right side
- drag files from left to right to upload them / from right to left to download them
- double-click on files in the left side to upload them / in the right side to download them
- if you've created/deleted/moved files/directories, you might have to press F5 to reload the current directory