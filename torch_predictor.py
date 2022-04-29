# Imports
import torch
from tqdm import tqdm

from factory import Factory
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from data_handling.dataloader_torch import TorchDataLoader
from trainers.u_net import UNetTrainer
import tensorflow.keras as K
from utils import *


# Fixed constants
offset = 144  # Numbering of first test image
dataset = 'original'
segmentation_threshold = 0.5

# Parameters
model_name = 'unet'                                             # <<<<<<<<<<<<<<<<<< Insert model type
trained_model_path = 'cp_7.pt'                                  # <<<<<<<<<<<<<<<<<< Insert trained model name

# Create loader, trainer etc. from factory
factory = Factory.get_factory(model_name)
dataloader = factory.get_dataloader_class()(dataset=dataset)
model = factory.get_model_class()()
trainer = factory.get_trainer_class()(dataloader=dataloader, model=model)

# Load the trained model weights
model_data = torch.load(trained_model_path)
model.load_state_dict(model_data['model'])
model.eval()

preprocessing = trainer.preprocessing
test_loader = dataloader.get_unlabeled_testing_dataloader(batch_size=1, preprocessing=preprocessing)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

create_or_clean_directory(OUTPUT_PRED_DIR)

# Prediction
with torch.no_grad():
    for i, x in tqdm(enumerate(test_loader)):
        x = x.to(device, dtype=torch.float32)
        output = model(x)
        pred = (output >= segmentation_threshold).cpu().detach().numpy().astype(int) * 255
        while len(pred.shape) > 3:
            pred = pred[0]
        K.preprocessing.image.save_img(f'{OUTPUT_PRED_DIR}/satimage_{offset+i}.png', pred, data_format="channels_first")
        del x



