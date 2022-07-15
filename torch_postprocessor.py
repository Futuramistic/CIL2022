# Imports
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from models.learning_aerial_image_segmenation_from_online_maps.Unet import UNet
from data_handling.dataloader_torch import TorchDataLoader
from trainers.u_net import UNetTrainer


##########################################################################
################################  Note  ##################################
##########################################################################
# This is used to do augmented prediction, i.e. not predict only on the input image,
# but also on transformed versions, the predictions are then ensembled in the hope
# of getting better final predictions.
# Currently supported transforms are 90°-angle rotations as well as horizontal/vertical
# flips.
# Tests performed on the Unet so far show no improvement but rather a lower performance.
# This can be explained by the fact that the loader is loading training data, for which
# it has already trained. It has thus probably over-fitted a bit on the training data
# and hence achieves better performance when only predicting on it and achieves way worse
# performance when predicting on rotated/flipped versions. This means that the training set
# should contain more augmented images.

# This can thus be used as a good and simple benchmark to measure over-fitting.


# Parameters

model = DeepLab()
trained_model_path = 'cp_final.pt'
dataset = 'original'


# Transformation functions
def rotation_transform(image, angle, inverse=False):
    if not inverse:
        image = TF.rotate(image, angle)
    else:
        image = TF.rotate(image, -angle)
    return image


def v_flip_transform(image):
    image = TF.vflip(image)
    return image


def h_flip_transform(image):
    image = TF.hflip(image)
    return image


def _augment(image):
    """
    Augment the input with the 90° rotated versions and the horizontal and vertical flipped versions
    Args:
        image (Tensor [1, ..., H, W])
    Returns:
        augmented tensor (Tensor [6, ..., H, W])
    """
    xs = []
    for i in range(4):
        xs.append(rotation_transform(image, i * 90, inverse=False))
    flipped = v_flip_transform(image)
    for i in range(4):
        xs.append(rotation_transform(flipped, i * 90, inverse=False))
    # xs.append(v_flip_transform(image))
    # xs.append(h_flip_transform(image))
    augmented = torch.cat(xs, dim=0)
    return augmented


def _unify(images):
    """
    Perform the inverse operation from augment. First the inverse transforms are applied,
    then the predictions are ensembled into one image.
    Args:
        images (Tensor [6, ..., H, W])
    Returns:
        a single ensembled image (Tensor [1, ..., H, W])
    """
    ll = []
    for i in range(images.shape[0]):
        img_i = torch.unsqueeze(rotation_transform(images[i], i * 90, inverse=True), dim=0)
        ll.append(img_i)
    for i in range(images.shape[0]):
        img_i = torch.unsqueeze(rotation_transform(images[i+4], i * 90, inverse=True), dim=0)
        ll.append(torch.unsqueeze(v_flip_transform(img_i), dim=0))
    # ll.append(torch.unsqueeze(v_flip_transform(images[4]), dim=0))
    # ll.append(torch.unsqueeze(v_flip_transform(images[5]), dim=0))
    images = torch.cat(ll, dim=0)
    return _ensemble(images)


def _ensemble(images):
    """
    Average the input images along the first dimension
    Args:
        images (Tensor[..., ..., H, W])
    Returns:
        an ensembled image (Tensor[1, ..., H, W])
    """
    ensembled = torch.mean(images, dim=0, keepdim=True)
    return ensembled


# Load the model/data
data = torch.load(trained_model_path)
model.load_state_dict(data['model'])
model.eval()

dataloader = TorchDataLoader(dataset=dataset)
trainer = UNetTrainer(dataloader, model)
preprocessing = trainer.preprocessing
loss_fn = trainer.loss_function

test_loader = dataloader.get_testing_dataloader(batch_size=1, preprocessing=preprocessing)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Transformed/Augmented prediction
test_loss = 0
test_loss_aug = 0
with torch.no_grad():
    for (x, y) in tqdm(test_loader):
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        preds = model(x)
        loss = loss_fn(preds, y).item()
        test_loss += loss
        x = _augment(x)
        preds = model(x)
        preds = _unify(preds)
        loss = loss_fn(preds, y).item()
        test_loss_aug += loss
        del x
        del y

# Print the losses
test_loss /= len(test_loader.dataset)
test_loss_aug /= len(test_loader.dataset)
print(f'\nTest loss: {test_loss:.3f}')
print(f'\nTest loss Aug: {test_loss_aug:.3f}')

