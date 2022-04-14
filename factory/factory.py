import abc

from trainers.trainer_torch import TorchTrainer
from trainers.trainer_tf import TFTrainer

from trainers.u_net import UNetTrainer
from trainers.UnetTF import UNetTFTrainer
from trainers.UnetPlusPlusTF import UNetPlusPlusTrainer
from trainers.AttUnetTF import AttUNetTrainer
from trainers.AttUnetPlusPlusTF import AttUNetPlusPlusTrainer
from trainers.gl_dense_u_net import GLDenseUNetTrainer

from models.TF.AttUNetTF import AttUnetTF
from models.TF.AttUNetPlusPlusTF import AttUNetPlusPlusTF

from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from models.TF.UNetTF import UNetTF
from models.TF.UNetPlusPlusTF import UNetPlusPlusTF
from models.road_extraction_from_high_res_rsi_using_dl.gl_dense_u_net import GLDenseUNet
from models.learning_aerial_image_segmenation_from_online_maps.Deeplabv3 import Deeplabv3

from data_handling.dataloader_tf import TFDataLoader
from data_handling.dataloader_torch import TorchDataLoader

from utils import MODEL_CLASS_DICT


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
        else:
            if next(filter(lambda m: m.lower() == model_name_lower_no_sep, MODEL_CLASS_DICT), None) is not None:
                raise NotImplementedError(f"The factory for the model {model_name} doesn't exist yet. "
                                          f"Check for Implementation in factory.factory.py")
            else:
                print("Check if you wrote the model name correctly and added the model to utils.MODEL_CLASS_DICT.")


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
        return TorchTrainer

    def get_model_class(self):
        return Deeplabv3

    def get_dataloader_class(self):
        return TorchDataLoader
