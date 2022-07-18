from data_handling import *
from models import *
from models.reinforcement.first_try import SimpleRLCNNMinimal
from trainers import *
from trainers.rl_trainer_torch_minimal import TorchRLTrainerMinimal


class Factory(abc.ABC):
    """
    Abstract class for the factory method, in order to create corresponding trainer and dataloader for a specific model.
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
        elif model_name_lower_no_sep in ["2shotnet", "twoshotnet"]:
            return TwoShotNetFactory()
        elif model_name_lower_no_sep in ["deeplabv3gan", "deeplabv3plusgan"]:
            return DeepLabV3PlusGANFactory()
        elif model_name_lower_no_sep in ['unetexptf','unetexp']:
            return UNetExpFactory()
        elif model_name_lower_no_sep in ['unet3plus','unet3+']:
            return UNet3PlusFactory()
        elif model_name_lower_no_sep == "simplerlcnn":
            return SimpleRLCNNFactory()
        elif model_name_lower_no_sep == "simplerlcnnminimal":
            return SimpleRLCNNMinimalFactory()
        elif model_name_lower_no_sep == "fftunet":
            return FFT_UNetFactory()
        elif model_name_lower_no_sep == "segformer":
            return SegFormerFactory()
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


class TwoShotNetFactory(Factory):
    def get_trainer_class(self):
        return UNetTrainer

    def get_model_class(self):
        return TwoShotNet
        
    def get_dataloader_class(self):
        return TorchDataLoader
        
        
class SimpleRLCNNFactory(Factory):
    def get_trainer_class(self):
        return TorchRLTrainer

    def get_model_class(self):
        return SimpleRLCNN

    def get_dataloader_class(self):
        return TorchDataLoader


class DeepLabV3PlusGANFactory(Factory):
    def get_trainer_class(self):
        return DeepLabV3PlusGANTrainer

    def get_model_class(self):
        return DeepLabV3PlusGAN
        
    def get_dataloader_class(self):
        return TorchDataLoader
        

class SimpleRLCNNMinimalFactory(Factory):
    def get_trainer_class(self):
        return TorchRLTrainerMinimal

    def get_model_class(self):
        return SimpleRLCNNMinimal

    def get_dataloader_class(self):
        return TorchDataLoader


class UNetExpFactory(Factory):
    def get_trainer_class(self):
        return UNetTFTrainer

    def get_model_class(self):
        return UNetExpTF

    def get_dataloader_class(self):
        return TFDataLoader


class UNet3PlusFactory(Factory):
    def get_trainer_class(self):
        return UNetTFTrainer

    def get_model_class(self):
        return UNet3PlusTF

    def get_dataloader_class(self):
        return TFDataLoader


class FFT_UNetFactory(Factory):
    def get_trainer_class(self):
        return FFT_UNetTrainer

    def get_model_class(self):
        return FFT_UNet

    def get_dataloader_class(self):
        return TorchDataLoader


class SegFormerFactory(Factory):
    def get_trainer_class(self):
        return SegFormerTrainer

    def get_model_class(self):
        return SegFormer

    def get_dataloader_class(self):
        return TorchDataLoader


def get_torch_scheduler(optimizer, scheduler_name, kwargs):
    """
    Get a Torch scheduler object, given its name
    Args:
        optimizer: The optimizer we want to schedule
        scheduler_name (str): The scheduler name
        kwargs: the arguments we want to pass to the created scheduler
    Returns:
        A learning rate scheduler
    """
    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, **kwargs)
    elif scheduler_name == 'lambdalr':
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, **kwargs)
    elif scheduler_name == 'lambdaevallr':
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=eval(kwargs['lr_lambda']),
                                                 **{k: v for k, v in kwargs.items() if k != 'lr_lambda'})
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **kwargs)
    else:
        # standard configuration
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)