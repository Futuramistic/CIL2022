from collections import namedtuple, deque
import functools
import random
from requests import head, patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal.windows import gaussian

class ReplayMemory(object):
    # inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, capacity, rl_network):
        self.memory = deque([],maxlen=capacity)
        self.rl_network = rl_network

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.rl_network.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RefinementQ(nn.Module):
    def __init__(self, num_conv_layer, patch_size=10):
        super(RefinementQ, self).__init__()
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        layers = []
        in_channels = 7 # RGB, Binary, 3 for canny (gradient magnitude, orientation, thin edges)
        curr_output_dim = patch_size
        kernel_size = 3
        stride=2
        padding=1
        for i in range(num_conv_layer): # construct like unet
            if i<num_conv_layer//2:
                layers.append(nn.Conv2d(in_channels, in_channels*2, kernel_size, stride, padding))
                in_channels *= 2
                layers.append(nn.BatchNorm2d(in_channels))
                layers.append(nn.LeakyReLU())
                curr_output_dim=((curr_output_dim - (kernel_size - 1) - 1) // stride  + 1)
            elif i>num_conv_layer//2 and i<num_conv_layer-1:
                layers.append(nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size, stride, padding))
                in_channels = in_channels//2
                layers.append(nn.BatchNorm2d(in_channels))
                layers.append(nn.LeakyReLU())
                curr_output_dim=((curr_output_dim - (kernel_size - 1) - 1) // stride  + 1)
            else:
                # final conv, can't have too large dimensions because of final linear layer
                layers.append(nn.Conv2d(in_channels, 1, kernel_size, stride=1, padding=1))
                layers.append(nn.Sigmoid()) # TODO sigmoid the best option here?
                curr_output_dim=((curr_output_dim - (kernel_size - 1) - 1)  + 1)
        self.conv_layers = nn.Sequential(*layers)
        self.canny = Canny()
        self.final_prediction = nn.Linear(curr_output_dim**2, 3) # output dim of 3 for velocity, angle, brush or don't brush
    
    def forward(self, x):
        edge_info = self.canny(x[:, 0:2]) # only RGB
        conv_in = torch.concat((*edge_info, x), dim=1) # stack in channel dim
        conv_out = self.conv_layers(conv_in)
        linear_in = conv_out.flatten(start_dim=1)
        linear_out = self.final_prediction(linear_in)
        vel_angle = F.tanh(linear_out[:,0:1]) # normalize velocity, angle
        brush = torch.round(F.sigmoid(linear_out))
        return torch.concat((vel_angle, brush), dim=1)

class SimpleRLCNN(nn.Module):
    def __init__(self, patch_size, in_channels=10):
        super(SimpleRLCNN, self).__init__()
        layers = []
        curr_output_dims = patch_size
        kernel_size = 3
        stride=1
        padding=1
        # initial input: RGB (3), history (5 by default), brush state (1)
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels, in_channels*2, kernel_size, stride, padding))
            in_channels *= 2
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.LeakyReLU())
            curr_output_dims=[((curr_output_dims[dim] - kernel_size + 2 * padding ) // stride  + 1) for dim in range(len(curr_output_dims))]
            
        self.convs = nn.Sequential(*layers)
        flattened_dims = functools.reduce(lambda x, y: x*y, curr_output_dims)*in_channels
        self.head = nn.Linear(flattened_dims, 5)
        
    # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate', for which we return the mean of the beta distribution
    # the action is then sampled in the trainer from that distribution --> we only need values between 0 and 1 --> sigmoid everything
    def forward(self, x):
        conv_out = self.convs(x)
        head_in = torch.flatten(conv_out, start_dim=1)
        return F.sigmoid(self.head(head_in))

# adapted from https://github.com/DCurro/CannyEdgePytorch to process batches
class Canny(nn.Module):
    def __init__(self, threshold=10.0):
        super(Canny, self).__init__()

        self.threshold = threshold

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])
        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])
        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        # torch.autograd.set_detect_anomaly(True)
        img_r = img[:,0:1] # [batch_size, 1, 224, 224]
        img_g = img[:,1:2]
        img_b = img[:,2:3]
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)
        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)
        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_mag = grad_mag.detach()
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0
        grad_orientation = grad_orientation
        # THIN EDGES (NON-MAX SUPPRESSION)
        all_filtered = self.directional_filter(grad_mag)
        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8
        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        batch_size = inidices_positive.size()[0]
        pixel_count = height * width * batch_size
        pixel_range = torch.FloatTensor([range(pixel_count)])
        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(batch_size, 1,height,width)
        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(batch_size,1,height,width)
        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])
        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0
        return grad_mag.detach(), grad_orientation.detach(), thin_edges.detach()
