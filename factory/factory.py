import abc

from data_handling import *
from models import *
from trainers import *
from utils import *


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
        model_name_lower_no_sep = model_name.lower().replace('-', '').replace('_', '')
        if model_name_lower_no_sep == "unet":
            return UNetFactory()
        elif model_name_lower_no_sep in ["unettf", "unettensorflow"]:
            return UNetTFFactory()
        elif model_name_lower_no_sep in ["unet++", "unetplusplus"]:
            return UNetPlusPlusFactory()
        elif model_name_lower_no_sep == "attunet":
            return AttUNetFactory()
        elif model_name_lower_no_sep in ["attunet++", "attentionunet++", "attentionunetplusplus", "attunetplusplus",
                                         "attunetplusplustf"]:
            return AttUNetPlusPlusTFFactory()
        elif model_name_lower_no_sep == "gldenseunet":
            return GLDenseUNetFactory()
        elif model_name_lower_no_sep == "deeplabv3":
            return DeepLabV3Factory()
        elif model_name_lower_no_sep == "cranet":
            return CRANetFactory()
        else:
            print(f"The factory for the model {model_name} doesn't exist. Check if you wrote the model name "
                  f"correctly and implemented a corresponding factory in factory.py.")


class UNetFactory(Factory):
    def get_trainer_class(self):
        return UNetTrainer

    def get_model_class(self):
        return UNet

    def get_dataloader_class(self):
        return TorchDataLoader


class UNetTFFactory(Factory):
    def get_trainer_class(self):
        return UNetTFTrainer

    def get_model_class(self):
        return UNetTF

    def get_dataloader_class(self):
        return TFDataLoader


class UNetPlusPlusFactory(Factory):
    def get_trainer_class(self):
        return UNetPlusPlusTrainer

    def get_model_class(self):
        return UNetPlusPlusTF

    def get_dataloader_class(self):
        return TFDataLoader


class AttUNetFactory(Factory):
    def get_trainer_class(self):
        return AttUNetTrainer

    def get_model_class(self):
        return AttUnetTF

    def get_dataloader_class(self):
        return TFDataLoader


class AttUNetPlusPlusTFFactory(Factory):
    def get_trainer_class(self):
        return AttUNetPlusPlusTrainer

    def get_model_class(self):
        return AttUNetPlusPlusTF

    def get_dataloader_class(self):
        return TFDataLoader


class GLDenseUNetFactory(Factory):
    def get_trainer_class(self):
        return GLDenseUNetTrainer

    def get_model_class(self):
        return GLDenseUNet

    def get_dataloader_class(self):
        return TFDataLoader


class DeepLabV3Factory(Factory):
    def get_trainer_class(self):
        return DeepLabV3Trainer

    def get_model_class(self):
        return Deeplabv3

    def get_dataloader_class(self):
        return TorchDataLoader


class CRANetFactory(Factory):
    def get_trainer_class(self):
        return CRANetTrainer

    def get_model_class(self):
        return OurDinkNet50

    def get_dataloader_class(self):
        return TorchDataLoader
