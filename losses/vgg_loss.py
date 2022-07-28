import torch
from torchvision import models
from utils import to_cuda

class VGGPerceptualLoss(torch.nn.Module):
    """
    Compute the VGG Perceptual Loss
    In a nutshell, feed both the input and the targets through the VGG encoder and compute the L1 distance
    between the 2 outputs
    """
    def __init__(self, resize=False):
        """
        Args:
            resize: Whether to resize the input images or not
        """
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(to_cuda(models.vgg16(pretrained=True).features[:4].eval()))
        blocks.append(to_cuda(models.vgg16(pretrained=True).features[4:9].eval()))
        blocks.append(to_cuda(models.vgg16(pretrained=True).features[9:16].eval()))
        blocks.append(to_cuda(models.vgg16(pretrained=True).features[16:23].eval()))
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(to_cuda(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)))
        self.std = torch.nn.Parameter(to_cuda(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 1)
            target = torch.unsqueeze(target, 1)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = to_cuda((input-self.mean) / self.std)
        target = to_cuda((target-self.mean) / self.std)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
