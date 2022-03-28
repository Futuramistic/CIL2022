import abc
from utils import MODEL_CLASS_DICT

from trainers.u_net import UNetTrainer
from trainers.gl_dense_u_net import GLDenseUNetTrainer
from trainers.trainer_torch import TorchTrainer
from trainers.trainer_tf import TFTrainer

from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from models.road_extraction_from_high_res_rsi_using_dl.gl_dense_u_net import GLDenseUNet
from models.learning_aerial_image_segmenation_from_online_maps.Deeplabv3 import Deeplabv3

from data_handling.dataloader_tf import TFDataLoader
from data_handling.dataloader_torch import TorchDataLoader

class Factory(abc.ABC):
    """Abstract class for the factory method, in order to create corresponding trainer and dataloader for a specific model.
    Use the static method "get_factory(model_name: string) to get the corresponding factory class
    """
    
    @abc.abstractmethod
    def get_trainer_class(self):
        return NotImplementedError
    
    @abc.abstractmethod
    def get_model_class(self):
        return NotImplementedError
    
    @abc.abstractmethod
    def get_dataloader_class(self):
        return NotImplementedError
    
    @staticmethod
    def get_factory(model_name):
        if model_name == "unet":
            return UNetFactory()
        elif model_name == "deeplabv3":
            return DeepLabV3Factory()
        elif model_name == "GLDenseUnet":
            return GLDenseUnetFactory()
        else:
            try:
                MODEL_CLASS_DICT[model_name]
                raise NotImplementedError(f"The factory for the model {model_name} doesn't extist yet. Check for Implementation in factory.factory.py")
            except KeyError:
                print("Check if you wrote the model name correctly and added the model to utils.MODEL_CLASS_DICT.")
                return

class UNetFactory(Factory):
    def get_trainer_class(self):
        return UNetTrainer
    def get_model_class(self):
        return UNet
    def get_dataloader_class(self):
        return TorchDataLoader

class GLDenseUnetFactory(Factory):
    def get_trainer_class(self):
        return GLDenseUNetTrainer
    def get_model_class(self):
        return GLDenseUNet
    def get_dataloader_class(self):
        return TFDataLoader

class DeepLabV3Factory(Factory):
    def get_trainer_class(self):
        return TorchTrainer #TODO: adapt if specific trainer implemented (if needed)
    def get_model_class(self):
        return Deeplabv3
    def get_dataloader_class(self):
        return TorchDataLoader
