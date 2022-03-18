import abc
import datetime, os, shutil, urllib3, zipfile
import warnings
import errno
from utils import ROOT_DIR, DATASET_ZIP_URLS


class DataLoader(abc.ABC):
    """Args:
            dataset (string): type of Dataset ("original" [from CIL2022 Competition], "Massachusets")
        """
    def __init__(self, dataset = "original"):
        self.dataset = dataset
        check = self.__download_data(dataset)
        if check == -1:
            print("Using default dataset from the CIL 2022 competition")
            self.dataset = "original"
        
        # set data directories
        self.img_dir = f'dataset/{dataset}/training/images'
        self.gt_dir = f'dataset/{dataset}/training/groundtruth'
        self.test_img_dir = f'dataset/{dataset}/test/images'
        self.test_gt_dir = None
        # some data sets have ground truth for their testing data:
        if os.path.exists(f"{ROOT_DIR}/dataset/{dataset}/test/groundtruth"):
            self.test_gt_dir = f'dataset/{dataset}/test/groundtruth'
        
        # define dataset variables for later usage
        self.training_data = None
        self.testing_data = None
        self.unlabeled_testing_data = None
       
    @abc.abstractmethod
    def get_training_dataloader(self, split, batch_size, **args):
        """
        Args:
            split (float): training/testing splitting ratio \in [0,1]
            batch_size (int): training batch size
            **args:for tensorflow e.g. 
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')
    
    @abc.abstractmethod
    def get_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args:for tensorflow e.g. 
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')
    
    @abc.abstractmethod
    def get_unlabeled_testing_dataloader(self, batch_size, **args):
        """
        Args:
            batch_size (int): training batch size
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')
    
    @abc.abstractmethod
    def load_model(self, path):
        raise NotImplementedError('must be defined for torch or tensorflow loader')
    
    @abc.abstractmethod
    def save_model(self, model, path):
        raise NotImplementedError('must be defined for torch or tensorflow loader')
    
    def __download_data(self, dataset_name):
        destination_path = os.path.join(*[ROOT_DIR, "dataset", dataset_name.lower()])
        ts_path = os.path.join(destination_path, "download_timestamp.txt")
        zip_path = f"{destination_path}.zip"

        url = next((v for k, v in DATASET_ZIP_URLS.items() if dataset_name.lower() == k.lower()), None)
        if url is None:
            warnings.warn(f"Dataset '{dataset_name}' unknown... error in Dataloader.__download_data()")
            return -1

        # check if data already downloaded; use timestamp file written *after* successful download for the check
        if os.path.exists(ts_path):
            return 1
        else:
            os.makedirs(destination_path, exist_ok=True)
        
        # data doesn't exist yet
        print("Downloading Dataset...")
        pool = urllib3.PoolManager()
        try:
            with pool.request("GET", url, preload_content=False) as response, open(zip_path, "wb") as file:
                shutil.copyfileobj(response, file)
        except Exception as e:
            warnings.warn(f"Error encountered while downloading dataset '{dataset_name}': {str(e)}")
            return -1
        print("...Done!")

        print("Extracting files...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(destination_path)
            print("...Done!")
        print("Removing zip file...")
        os.unlink(zip_path)
        print("...Done!")

        with open(ts_path, "w") as file:
            file.write(str(datetime.datetime.now()))
        
        return 1
    
    def __create_file_structure(self, destination_path):
        # create paths 
        # --> useful if dataset is not in required format, currently not needed anymore
        #####TODO: test this method#####
        try:
            os.makedirs(destination_path)
            os.makedirs(f"{destination_path}/training")
            os.makedirs(f"{destination_path}/training/images")
            os.makedirs(f"{destination_path}/training/groundtruth")
            os.makedirs(f"{destination_path}/test")
            os.makedirs(f"{destination_path}/test/images")
            if "test_gt_path_initial" in locals():
                os.makedirs(f"{destination_path}/test/groundtruth")
            os.makedirs(f"{destination_path}/initial")
            return 1
        except OSError as oe:
            # data already exists
            if oe.errno == errno.EEXIST:
                return 1
        except:
            warnings.warn(f"Ooops, there has been an error while creating the folder structure in {destination_path} in the Dataloader")
            return -1
    
    def __move_files(self, destination_path, image_path_initial, gt_path_initial, test_image_path_initial, test_gt_path_initial):
        # move all files into the wanted folder structure 
        # --> might be useful later for other datasets, currently not needed anymore
        #####TODO: test this method#####
        print("Moving Data into required folder structure...")
        for file_name in os.listdir(image_path_initial):
            shutil.move(src = os.path.join(image_path_initial,file_name), dst = f"{destination_path}/training/images")
        for file_name in os.listdir(gt_path_initial):
            shutil.move(src = os.path.join(gt_path_initial, file_name), dst = f"{destination_path}/training/groundtruth")
        for file_name in os.listdir(test_image_path_initial):
            shutil.move(src = os.path.join(test_image_path_initial, file_name), dst = f"{destination_path}/test/images")
        if "test_gt_path_initial" in locals():
            for file_name in os.listdir(test_gt_path_initial):
                shutil.move(src = os.path.join(test_gt_path_initial, file_name), dst = f"{destination_path}/test/groundtruth")
        # remove initial files
        print("Removing old file structure from download...")
        shutil.rmtree(f"{destination_path}/initial")
        print("...Done")