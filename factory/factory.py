import abc
from models.TF.AttUNetTF import AttUnetTF
from trainers.AttUnetTF import AttUNetTrainer
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
        model_name_lower = model_name.lower()
        if model_name_lower == "unet":
            return UNetFactory()
        elif model_name_lower == "deeplabv3":
            return DeepLabV3Factory()
        elif model_name_lower in ["gldenseunet", "gl-dense-u-net"]:
            return GLDenseUnetFactory()
        elif model_name_lower in ["attunet", "att-u-net", "att_unet"]:
            return AttUNetFactory()
        else:
            if next(filter(lambda m: m.lower() == model_name_lower, MODEL_CLASS_DICT), None) is not None:
                raise NotImplementedError(f"The factory for the model {model_name} doesn't exist yet. Check for Implementation in factory.factory.py")
            else:
                print("Check if you wrote the model name correctly and added the model to utils.MODEL_CLASS_DICT.")

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

class AttUNetFactory(Factory):
    def get_trainer_class(self):
        return AttUNetTrainer #TODO: adapt if specific trainer implemented (if needed)
    def get_model_class(self):
        return AttUnetTF
    def get_dataloader_class(self):
        return TFDataLoader