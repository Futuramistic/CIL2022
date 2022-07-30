from collections import namedtuple, deque
import functools
import random
from typing import Optional, Callable, Type, Union, List
from requests import head, patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from scipy.signal.windows import gaussian
from torchvision.models import resnet50


class RefinementQ(nn.Module):
    """The Refinement-Network as a try to use Reinforcement Learning to refine other models segmentation predictions.
    Due to lacking performance of the simpler networks, the refinement q-network was never tested excessively but 
    can be used in future tries.
    Args:
        num_conv_layer (int): the depth of the refinement network
        patch_size (int): the length of the quadratic patch. Needed to compute the dimensions for the final linear
        layer.
    """
    def __init__(self, num_conv_layer, patch_size=10):
        super(RefinementQ, self).__init__()
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        layers = []
        in_channels = 7 # RGB, Binary, 3 for canny (gradient magnitude, orientation, thin edges) TODO: make in_channels dynamic to history size
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
        brush = torch.round(torch.sigmoid(linear_out))
        return torch.concat((vel_angle, brush), dim=1)


class SimpleRLCNN(nn.Module):
    """Simple Reinforcement Learning Convolutional Neural Network
    Allthough this model does not achieve a very well performance (most likely to the inefficient trajectory sampling
    as well as unsuited penalties), it's main ideas are a new approach to image segmentation and can 
    be refined in the future. In the main idea, the agent is given the opportunity to segment a given patch. The agent
    is set at the center of the patch and can either put down the brush to segment selected pixels as road by determining
    delta angle, radius and magnitude of the brush. The brush is a "circle" which paints all pixels within the circle.
    This allows for the usually round ends of a street to be painted easily.
    
    The CNN takes as input a history of the last actions, the new patch to segment, and optionally other information
    concatenated to the input, such as e.g. the last brush state or a minimap of the agent's current position. 
    It outputs actions, which are atm 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate'.

    Args:
        patch_size (int, int): the size of the observations for the actor. Defaults to (10,10).
        in_channels (int): The number of input channels depending on the history size. Defaults to 10.
    """
    def __init__(self, patch_size=(10,10), in_channels=10):
        super(SimpleRLCNN, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        
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
        return torch.sigmoid(self.head(head_in))


class SimpleRLCNNMinimal(nn.Module):

    ##############################################################################
    # TODO: adapt description
    ##############################################################################


    """Simple Reinforcement Learning Convolutional Neural Network - Minimal Version
    Because the non-minimal model does not achieve a very good performance, a minimal solution is proposed which for which this
    model architecture is used in the trainer.
    It outputs actions, which are atm 'delta_angle', 'brush_state' in non-supervised settings, and
    'angle', 'brush_radius', 'magnitude' in supervised settings.

    Args:
        patch_size (int, int): the size of the observations for the actor. Defaults to (10, 10).
        in_channels (int): The number of input channels. Defaults to 5.
        out_channels (int): The number of output channels. Defaults to 3 for the supervised setting,
                            and 2 for the unsupervised one.
    """
    def __init__(self, patch_size=(10, 10), in_channels=5, out_channels=2):
        super(SimpleRLCNNMinimal, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        self.head = nn.Linear(flattened_dims, self.out_channels) # brush state and delta_angle
    
    ##############################################################################
    # TODO: adapt comment
    ##############################################################################

    # action is: 'delta_angle', 'magnitude', 'brush_state', 'brush_radius', 'terminate', for which we return the mean of the beta distribution
    # the action is then sampled in the trainer from that distribution --> we only need values between 0 and 1 --> sigmoid everything
    def forward(self, x):
        x = x.float()
        conv_out = self.convs(x)
        head_in = torch.flatten(conv_out, start_dim=1)
        return torch.sigmoid(self.head(head_in))


class SimpleRLCNNMinimalSupervised(SimpleRLCNNMinimal):
    # simply uses different default values than SimpleRLCNNMinimal
    def __init__(self, patch_size, in_channels=5, out_channels=3):
        super(SimpleRLCNNMinimalSupervised, self).__init__(patch_size, in_channels, out_channels)


class ResNetBasedRegressor(nn.Module):
    def __init__(self, patch_size):
        super(ResNetBasedRegressor, self).__init__()
        self.patch_size = patch_size
        # ResNet34: BasicBlock, [3, 4, 6, 3]
        self.regressor = ResNetRegressor(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        return self.regressor(x)

class Canny(nn.Module):
    """Canny torch version in order to incorporate Canny edge detection easily into the models for faster
    computation
    adapted from https://github.com/DCurro/CannyEdgePytorch to process batches
    """
    def __init__(self):
        super(Canny, self).__init__()

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



# code taken from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py begins here
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetRegressor(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_in_channels=5,
        num_out_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_in_channels = num_in_channels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.num_in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.sigmoid(x)

    def forward(self, x: Tensor):
        return self._forward_impl(x)

