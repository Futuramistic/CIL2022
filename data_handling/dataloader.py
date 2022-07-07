import abc
import datetime, os, shutil, urllib3, zipfile
import warnings
import errno
import numpy as np
from utils import ACCEPTED_IMAGE_EXTENSIONS, DATASET_ZIP_URLS, ROOT_DIR


class DataLoader(abc.ABC):
    """Args:
            dataset (string): type of Dataset ("original" [from CIL2022 Competition], "Massachusets")
        """

    def __init__(self, dataset="original"):
        self.dataset = dataset
        check = self.__download_data(dataset)
        if check == -1:
            print("Using default dataset from the CIL 2022 competition")
            self.dataset = "original"

        # set data directories
        ds_base = [ROOT_DIR, "dataset", dataset]
        self.training_img_dir = os.path.join(*[*ds_base, "training", "images"])
        self.training_gt_dir = os.path.join(*[*ds_base, "training", "groundtruth"])
        self.test_img_dir = os.path.join(*[*ds_base, "test", "images"])
        self.test_gt_dir = None
        # some data sets have ground truth for their test data:
        test_gt_dir = os.path.join(*[*ds_base, "test", "groundtruth"])
        if os.path.exists(test_gt_dir):
            self.test_gt_dir = test_gt_dir

        self.training_img_paths, self.training_gt_paths = DataLoader.get_img_gt_paths(self.training_img_dir, self.training_gt_dir)
        self.test_img_paths, self.test_gt_paths = DataLoader.get_img_gt_paths(self.test_img_dir, self.test_gt_dir, initial_shuffle=False)

        # define dataset variables for later usage
        self.training_data = None
        self.testing_data = None
        self.unlabeled_testing_data = None
    
    
    @staticmethod
    def get_img_gt_paths(img_dir, gt_dir, initial_shuffle=True):
        img_paths, gt_paths = None, None
        img_idxs = None  # match corresponding image<>gt pairs
        have_samples = img_dir is not None
        have_gt = gt_dir is not None
        if have_samples:
            img_paths = []
            img_idxs = {}
            for img_name in sorted(os.listdir(img_dir)):
                _, ext = os.path.splitext(img_name)
                if ext.lower() in ACCEPTED_IMAGE_EXTENSIONS:
                    pth = f"{img_dir}/{img_name}"
                    img_idxs[img_name] = len(img_paths)
                    img_paths.append(pth)
        if have_gt:
            gt_paths = [""] * len(img_paths) if have_samples else []
            for img_name in os.listdir(gt_dir):
                if have_samples and img_name not in img_idxs:
                    continue
                _, ext = os.path.splitext(img_name)
                if ext.lower() in ACCEPTED_IMAGE_EXTENSIONS:
                    pth = f"{gt_dir}/{img_name}"
                    if have_samples:
                        gt_paths[img_idxs[img_name]] = pth
                    else:
                        gt_paths.append(pth)

        # Shuffle the dataset once to have more diversity when visualizing images
        if initial_shuffle:
            np.random.seed(42)
            shuffler = np.random.permutation(len(img_paths))
            img_paths = np.array(img_paths)[shuffler]
            if gt_paths is not None:
                gt_paths = np.array(gt_paths)[shuffler]

        return img_paths, gt_paths

    @abc.abstractmethod
    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/test splitting ratio \in [0,1]

        Returns:
            Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
            in samples
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_training_dataloader(self, split, batch_size, preprocessing=None, **args):
        """
        Args:
            split (float): training/test splitting ratio \in [0,1]
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args:for tensorflow e.g.
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args:for tensorflow e.g.
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    def get_default_evaluation_interval(self, split, batch_size, num_epochs, num_samples_to_visualize):
        train_dataset_size, test_dataset_size, _ = self.get_dataset_sizes(split)
        iterations_per_epoch = train_dataset_size // batch_size
        # every time EVALUATE_AFTER_PROCESSING_SAMPLES samples are processed, perform an evaluation
        # cap frequency at one evaluation per MIN_EVALUATION_INTERVAL iterations
        EVALUATE_AFTER_PROCESSING_SAMPLES = 200
        MIN_EVALUATION_INTERVAL = 20
        interval = max(MIN_EVALUATION_INTERVAL, EVALUATE_AFTER_PROCESSING_SAMPLES // batch_size)
        return interval

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
            local_dataset_path = os.path.join(*[ROOT_DIR, "dataset", dataset_name.lower()])
            if os.path.exists(local_dataset_path):
                warnings.warn(f"Dataset '{dataset_name}' is a local dataset. Consider uploading it to polybox")
                return 1
            else:
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
            warnings.warn(
                f"Ooops, there has been an error while creating the folder structure in {destination_path} in the Dataloader")
            return -1

    def __move_files(self, destination_path, image_path_initial, gt_path_initial, test_image_path_initial,
                     test_gt_path_initial):
        # move all files into the wanted folder structure 
        # --> might be useful later for other datasets, currently not needed anymore
        #####TODO: test this method#####
        print("Moving Data into required folder structure...")
        for file_name in os.listdir(image_path_initial):
            shutil.move(src=os.path.join(image_path_initial, file_name), dst=f"{destination_path}/training/images")
        for file_name in os.listdir(gt_path_initial):
            shutil.move(src=os.path.join(gt_path_initial, file_name), dst=f"{destination_path}/training/groundtruth")
        for file_name in os.listdir(test_image_path_initial):
            shutil.move(src=os.path.join(test_image_path_initial, file_name), dst=f"{destination_path}/test/images")
        if "test_gt_path_initial" in locals():
            for file_name in os.listdir(test_gt_path_initial):
                shutil.move(src=os.path.join(test_gt_path_initial, file_name),
                            dst=f"{destination_path}/test/groundtruth")
        # remove initial files
        print("Removing old file structure from download...")
        shutil.rmtree(f"{destination_path}/initial")
        print("...Done")
