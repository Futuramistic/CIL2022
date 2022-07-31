# code adapted from https://github.com/yan-hao-tian/lawin and https://github.com/NVlabs/SegFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import hashlib
import urllib
import os

from mmcv.cnn import ConvModule, NonLocal2d
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import BaseModule, auto_fp16, force_fp32

# do not remove these imports, as these names need to be in score to construct the requested backbones
from .segformer import mit_bone, mit_btwo, mit_bthree, mit_bfour, mit_bfive


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """
    Resize an input tensor to the given size.
    Args:
        size (list of int): (H, W) of desired output, or None to use scale_factor
        scale_factor (float): scale factor of desired output, or None to use size
        mode (str): interpolation mode, as in torch.nn.functional.interpolate
        align_corners (bool): align_corners arg, as in torch.nn.functional.interpolate
        warning: whether to issue a warning if the output and input sizes are chosen suboptimally
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """
    Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.
    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
        """
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """
    def __init__(self, proj_type='pool', patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """
        Constructor
        Args:
            proj_type (str): type of projection ("conv" or "pool")
            patch_size (int): patch size
            in_chans (int): number of channels in input image
            embed_dim (int): dimension of embedding
            norm_layer (class): class to construct normalization layer
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, groups=patch_size*patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size), nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
        """
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
        if self.proj_type == 'conv': 
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class LawinAttn(NonLocal2d):
    """
    Lawin Attention Module: https://github.com/yan-hao-tian/lawin
    """
    def __init__(self, *arg, head=1, patch_size=None, **kwargs):
        """
        Constructor
        Args:
            head (int): number of heads
            patch_size (int): patch size
        """
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size
        
        if self.head!=1:
            self.position_mixing = nn.ModuleList([nn.Linear(patch_size*patch_size, patch_size*patch_size) for _ in range(self.head)])

    def forward(self, query, context):
        """
        Calculate the output for the input tensor (see Lawin paper)
        Args:
            query (torch.Tensor): query input tensor
            context (torch.Tensor): context input tensor
        """
        # x: [N, C, H, W]
        
        n = context.size(0)
        n, c, h, w = context.shape
        
        if self.head!=1:
            context = context.reshape(n, c, -1)
            context_mlp = []
            for hd in range(self.head):
                context_crt = context[:, (c//self.head)*(hd):(c//self.head)*(hd+1), :]
                context_mlp.append(self.position_mixing[hd](context_crt))

            context_mlp = torch.cat(context_mlp, dim=1)
            context = context+context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)


        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output


class BaseDecodeHead(BaseModule):
    """
    Base class for BaseDecodeHead.
    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        """
            Constructor
            Args:
                in_channels (int): number of input channels
                channels (int): channels of intermediate embedding
                num_classes (int): number of output classes
                dropout_ratio (float): dropout rate to apply to intermediate embedding
                conv_cfg (dict): dict with configuration for conv layers (mmsegmentation-style)
                norm_cfg (dict): dict with configuration for norm layers (mmsegmentation-style)
                norm_cfg (dict): dict with configuration for activation function (mmsegmentation-style)
                loss_decode (dict): ignored
                in_index (int|Sequence[int]): input feature index from backbone
                input_transform (str): transformation type of input features
                                       options: 'resize_concat', 'multiple_select', None.
                                                'resize_concat': Multiple feature maps will be resize to the
                                                                 same size as first one and than concat together.
                                                                 Usually used in FCN head of HRNet.
                ignore_index (int): ignored
                sampler (dict): ignored
                align_corners (bool): align_corners arg, as in torch.nn.functional.interpolate
                init_cfg (dict): dict with configuration for layer initialization (mmsegmentation-style)
        """
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.loss_decode = nn.CrossEntropyLoss()
        self.loss_decode.loss_name = 'CrossEntropyLoss'

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr"""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """
        Check and initialize input transforms
        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """
        Transform inputs for decoder
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    def forward(self, inputs):
        """Placeholder of forward function"""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """
        Forward function for training
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.
        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel"""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss"""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss


class LawinHead(BaseDecodeHead):
    """
    Lawin head: https://github.com/yan-hao-tian/lawin
    """

    def __init__(self, embed_dim=768, use_scale=True, reduction=2, **kwargs):
        """
        Constructor
        Args:
            embed_dim (int): dimension of intermediate embedding
            use_scale (bool): whether to use scaling for the underlying non-local module
                              (https://arxiv.org/abs/1711.07971)
            reduction (int): channel reduction ratio for the underlying non-local module
                             (https://arxiv.org/abs/1711.07971)
        """
        super(LawinHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvModule(512, 512, 1, conv_cfg=self.conv_cfg,
                                        norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.linear_c4 = MLP(input_dim=self.in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*3,
            out_channels=512,
            kernel_size=1
        )
        
        self.short_path = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1
        )
        
        self.cat = ConvModule(
            in_channels=512*5,
            out_channels=512,
            kernel_size=1
        )

        self.low_level_fuse = ConvModule(
            in_channels=560,
            out_channels=512,
            kernel_size=1
        )
        
        self.ds_8 = PatchEmbed(proj_type='pool', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='pool', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='pool', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)

    def get_context(self, x, patch_size):
        """
        Get context for spatial pyramid pooling (see Lawin paper)
        Args:
            x (torch.Tensor): input tensor
            patch_size (int): patch size
        """
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size*r, stride=patch_size, padding=int((r-1)/2*patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size*r, pw=patch_size*r, nh=h//patch_size, nw=w//patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context
    
    def forward(self, inputs):
        """
        Calculate the segmentation maps for the input features
        Args:
            inputs (torch.Tensor): input features from backbone
        """
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # (n, c, 32, 32)
        _c4 = resize(_c4, size=c2.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)

        output = []
        output.append(self.short_path(_c))
        output.append(resize(self.image_pool(_c),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=self.align_corners))

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
            output.append(_output)
        
        output = self.cat(torch.cat(output, dim=1))

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        output = resize(output, size=c1.size()[2:], mode='bilinear', align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))

        output = self.cls_seg(output)

        return output


class Lawin(nn.Module):
    """
    Lawin transformer: https://github.com/yan-hao-tian/lawin
    """
    def __init__(self, align_corners=False,
                 pretrained_backbone_path='https://polybox.ethz.ch/index.php/s/5j6rsXjTvcRWgSP/download',
                 backbone_name='mit_bfive'):
        """
        Constructor
        Args:
            align_corners (bool): align_corners argument of F.interpolate.
            pretrained_backbone_path (str): path to pretrained backbone (must match backbone_name)
            backbone_name (str): name of backbone to use (out of mit_bone, mit_btwo, mit_bthree, mit_bfour, mit_bfive)
                                 backbone paths:
                                 https://polybox.ethz.ch/index.php/s/Yj3EGcUlcMnqUgY/download for mit_b1
                                 https://polybox.ethz.ch/index.php/s/5j6rsXjTvcRWgSP for mit_b5
        """
        super(Lawin, self).__init__()
        if pretrained_backbone_path is not None and pretrained_backbone_path.lower().startswith('http'):
            os.makedirs('pretrained', exist_ok=True)
            url_hash = hashlib.md5(str.encode(pretrained_backbone_path)).hexdigest()
            target_path = os.path.join('pretrained', f'lawin_{url_hash}.pth')
            if not os.path.isfile(target_path):
                urllib.request.urlretrieve(pretrained_backbone_path, target_path)
            pretrained_backbone_path = target_path

        self.backbone = eval(backbone_name)(pretrained_backbone_path=pretrained_backbone_path)
        self.backbone_name = backbone_name

        self.head = LawinHead(in_channels=[64, 128, 320, 512], channels=512, num_classes=2,
                              in_index=[0, 1, 2, 3])

        self.align_corners = align_corners
        self.pretrained_backbone_path = pretrained_backbone_path

    def forward(self, x, apply_activation=True):
        """
        Calculate the segmentation maps for the input tensor
        Args:
            x (torch.Tensor): input tensor
            apply_activation (bool): whether to apply the activation function to the output of the network
                                     (default: True)
        """
        _, _, H, W = x.shape
        
        # resize input if necessary
        if H != 512 or W != 512:
            x = resize(x, (512, 512), mode='bilinear', align_corners=self.align_corners)
        feats = self.backbone(x)
        head_out = self.head(feats)
        _, _, H_H, H_W = head_out.shape
        
        # resize output if necessary
        if H_H != H or H_W != W:
            head_out = resize(head_out, (H, W), mode='bilinear', align_corners=self.align_corners)
        if apply_activation:
            head_out = F.softmax(head_out)
        return head_out
