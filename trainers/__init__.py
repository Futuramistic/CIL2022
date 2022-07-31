from .trainer import Trainer
from .trainer_tf import TFTrainer
from .trainer_torch import TorchTrainer

from .TF.attunetplusplus_tf_trainer import AttUNetPlusPlusTrainer
from .TF.gldenseunet_tf_trainer import GLDenseUNetTrainer
from .TF.unetplusplus_tf_trainer import UNetPlusPlusTrainer
from .TF.unet_tf_trainer import UNetTFTrainer
from .TF.u2net_tf_trainer import U2NetTFTrainer

from .Torch.unet_torch_trainer import UNetTrainer
from .Torch.cranet_torch_trainer import CRANetTrainer
from .Torch.deeplabv3_torch_trainer import DeepLabV3Trainer
from .Torch.deeplabv3gan_torch_trainer import DeepLabV3PlusGANTrainer
from .Torch.segformer_torch_trainer import SegFormerTrainer
from .Torch.lawin_torch_trainer import LawinTrainer

from .reinforcement.rl_trainer_torch import TorchRLTrainer
from .reinforcement.rl_trainer_torch_minimal import TorchRLTrainerMinimal
