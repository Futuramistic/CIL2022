import tensorflow as tf
import tensorflow.keras as K

from data_handling.dataloader_tf import TFDataLoader
from data_handling.dataloader_torch import TorchDataLoader
# from models.learning_aerial_image_segmenation_from_online_maps import DeepLabv3
from models.TF.UNetExpTF import UNetTF, UNetExpTF
from trainers.UnetTF import UNetTFTrainer
from trainers.deep_lab_v3 import DeepLabV3Trainer
from trainers.deeplabv3gan import DeepLabV3PlusGANTrainer
from losses.focalLoss import FocalLoss
from losses.focalTversky import FocalTverskyLoss
from losses.diceLoss import DiceLoss
from losses.diceBCELoss import BCELoss

import models
import trainers
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

model = models.cascade_residual_attention.CRA_Net.OurDinkNet50()
trained_model_path = 'cp_best_f1.pt'  # <<<<<<<<<<<<<< INSERT here
# Load the trained model weights
if torch.cuda.is_available():
    model_data = torch.load(trained_model_path)
else:
    model_data = torch.load(trained_model_path, map_location=device)
model.load_state_dict(model_data['model'])
model = model.to(device)
model.eval()
model_children = list(model.children())
print(model_children)
image = Image.open('dataset/original/training/images/satimage_22.png')
image = transform(image)
print(f"Image shape before: {image.shape}")
image = image[:3, :, :].unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

input = image
out, refined, outputs = model(input)
print('>>>>>>>>>', out.shape, refined.shape, len(outputs))
for element in outputs:
    print(element.shape)

print('Processing')
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    # a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')



# for child in model_children:
#     print(input.shape)
#     input = child(input)
#
# print('final shape', input['out'].shape)
#
# import torch.nn as nn
#
# # we will save the conv layer weights in this list
# model_weights =[]
# #we will save the 49 conv layers in this list
# conv_layers = []
# # get all the model children as list
# model_children = list(model.children())
# #counter to keep count of the conv layers
# counter = 0
# #append all the conv layers and their respective wights to the list
#
#
# def retrieve_conv_layers(model_):
#     global counter
#     for child in model_.children():
#         if type(child) == nn.Conv2d:
#             counter += 1
#             model_weights.append(child.weight)
#             conv_layers.append(child)
#         # elif type(child) == nn.Sequential:
#         #     for j in range(len(child)):
#         #         for child2 in child[j].children():
#         #             retrieve_conv_layers(child2)
#         #             # if type(child2) == nn.Conv2d:
#         #             #     counter += 1
#         #             #     model_weights.append(child.weight)
#         #             #     conv_layers.append(child)
#         else:
#             retrieve_conv_layers(child)
#
#
# retrieve_conv_layers(model)
#
#
# print(f"Total convolution layers: {counter}")
# print("conv_layers", conv_layers)
#
#
# outputs = []
# names = []
# for layer in conv_layers[0:]:
#     image = layer(image)
#     outputs.append(image)
#     names.append(str(layer))
# print(len(outputs))
#
# # print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)
