"""SegFormer Implementation
code taken from https://github.com/NVlabs/SegFormer/
Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
This work is licensed under the NVIDIA Source Code License"""

from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hashlib
import os
import urllib.request
import numpy as np
import warnings

from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


class Mlp(nn.Module):
    """
    MLP module with depthwise convolution: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Constructor
        Args:
            in_features: number of input features
            hidden_features: number of hidden layer features
            out_features: number of output layer features
            act_layer (nn.Module): activation to use for the hidden layer
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize all layers
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
            H (int): height of intermediate representation before depthwise convolution
            W (int): width of intermediate representation before depthwise convolution
        """
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Attention module: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        """
        Constructor
        Args:
            dim (int): dimension of Q, K and V
            num_heads (int): number of attention heads
            qkv_bias (bool): whether to use bias terms for Q, K and V
            qk_scale (float): normalizing weight to scale output of Q @ K by
            attn_drop (float): dropout rate to apply to softmax(Q @ K / norm)
            proj_drop (float): dropout rate to apply to projection of softmax(Q @ K / norm) @ V
            sr_ratio (int): ratio of spatial reduction attention (default: 1)
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the given module
        Args:
            m (nn.Module): module to initialize weights for
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
            H (int): height of intermediate representation before spatial reduction attention
            W (int): width of intermediate representation before spatial reduction attention
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer block: https://github.com/NVlabs/SegFormer/
    Args:
        dim (int): dimension of Q, K and V
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of intermediate MLP's embedding dim to "dim" parameter
        qkv_bias (bool): whether to use bias terms for Q, K and V
        qk_scale (float): normalizing weight to scale output of Q @ K by, or None to use default
        drop (float): dropout rate to apply to output of intermediate MLP and projection of
                        softmax(Q @ K / norm) @ V
        attn_drop (float): dropout rate to apply to softmax(Q @ K / norm)
        drop_path (float): path dropout rate
        sr_ratio (int): ratio of spatial reduction attention (default: 1)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
            H (int): height of intermediate representation before depthwise convolution
            W (int): width of intermediate representation before depthwise convolution
        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding: https://github.com/NVlabs/SegFormer/
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """
        Constructor
        Args:
            img_size (int): size of input image
            patch_size (int): patch size
            stride (int): stride to use
            in_chans (int): number of channels in the input image
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the given module
        Args:
            m (nn.Module): module to initialize weights for
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    """
    Mix vision transformer: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, img_size=400, patch_size=16, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pretrained_backbone_path=None):
        """
        Constructor
        Args:
            img_size (int): size of the input image
            patch_size (int): size of a patch
            in_chans (int): number of channels in the input image
            num_classes (int): number of output classes
            embed_dims (list of int): dimensions of embeddings for each block
            num_heads (list of int): number of attention heads for each block
            mlp_ratios (list of int): ratios of intermediate MLP's embedding dim to "dim" parameter for each block
            qkv_bias (bool): whether to use bias terms for Q, K and V
            qk_scale (float): normalizing weight to scale output of Q @ K by, or None to use default
            drop (float): dropout rate to apply to output of intermediate MLP and projection of
                softmax(Q @ K / norm) @ V
            attn_drop_rate (float): dropout rate to apply to softmax(Q @ K / norm)
            drop_path_rate (float): path dropout rate
            sr_ratios (list of int): list of ratios of spatial reduction attention for each block
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.pretrained_backbone_path = pretrained_backbone_path

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        if pretrained_backbone_path is not None:
            self.init_weights(pretrained=pretrained_backbone_path)

    def _init_weights(self, m):
        """
        Initialize the weights of the given module
        Args:
            m (nn.Module): module to initialize weights for
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """
        Initialize weights of the entire transformer
        Args:
            pretrained (str): path to file with pretrained weights, or None to skip loading pretrained weights
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=None)

    def reset_drop_path(self, drop_path_rate):
        """
        Reset the path dropout probabilities for all blocks
        Args:
            drop_path_rate (float): path dropout probability
        """
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Return names of all layers to which no weight decaying is applied
        """
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        """
        Return the classifier of this MiT
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        Reset the classifier of this MiT
        Args:
            num_classes (int): number of output classes
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Calculate the features for the input tensor
        Args:
            x (torch.Tensor): input tensor
        """
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        """
        Calculate the features for the input tensor
        Args:
            x (torch.Tensor): input tensor
        """
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    """
    Depth-wise convolution layer
    Args:
        dim (int): dimension of embedding
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
            H (int): height of intermediate representation before depthwise convolution
            W (int): width of intermediate representation before depthwise convolution
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# difference across mit backbones lies in depths and embedding dims

class mit_bzero(MixVisionTransformer):
    """
    MiT-B0 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_bzero, self).__init__(
            img_size=400, patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)

class mit_bone(MixVisionTransformer):
    """
    MiT-B1 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_bone, self).__init__(
            img_size=400, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)


class mit_btwo(MixVisionTransformer):
    """
    MiT-B2 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_btwo, self).__init__(
            img_size=400, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)


class mit_bthree(MixVisionTransformer):
    """
    MiT-B3 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_bthree, self).__init__(
            img_size=400, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)


class mit_bfour(MixVisionTransformer):
    """
    MiT-B4 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_bfour, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)

class mit_bfive(MixVisionTransformer):
    """
    MiT-B5 backbone: https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, pretrained_backbone_path=None, **kwargs):
        """
        Constructor
        Args:
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
        """
        super(mit_bfive, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained_backbone_path=pretrained_backbone_path)


# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

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


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        """
        Constructor
            input_dim (int): dimension of input
            embed_dim (int): dimension of embedding
        """
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


class SegFormerHead(nn.Module):
    """
    SegFormer head:  https://github.com/NVlabs/SegFormer/
    """
    def __init__(self, feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512], num_classes=2,
                 embedding_dim=256, dropout_rate=0.1):
        """
        Constructor
            feature_strides (list of int): ignored
            in_channels (list of int): number of input channels for each block
            num_classes (int): number of output classes
            embedding_dim (int): embedding dim for MLPs
            dropout_rate (float): droput
        """
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_strides = feature_strides
        self.dropout_rate = dropout_rate

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.embedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        Calculate the output for the input tensor
        Args:
            inputs (torch.Tensor): input tensor
        """
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, align_corners=False,
                 pretrained_backbone_path='https://polybox.ethz.ch/index.php/s/5j6rsXjTvcRWgSP/download',
                 backbone_name='mit_bfive'):
        """
        Constructor
        Args:
            align_corners (bool): align_corners arg for the output segmentation, as in torch.nn.functional.interpolate
            pretrained_backbone_path (str): path to backbone to load, or None to avoid loading a pretrained backbone
            backbone_name (str): name of backbone to use (out of mit_bone, mit_btwo, mit_bthree, mit_bfour, mit_bfive)
                                 backbone paths:
                                 https://polybox.ethz.ch/index.php/s/Yj3EGcUlcMnqUgY/download for mit_b1
                                 https://polybox.ethz.ch/index.php/s/5j6rsXjTvcRWgSP for mit_b5
        """
        super(SegFormer, self).__init__()
        if pretrained_backbone_path is not None and pretrained_backbone_path.lower().startswith('http'):
            os.makedirs('pretrained', exist_ok=True)
            url_hash = hashlib.md5(str.encode(pretrained_backbone_path)).hexdigest()
            target_path = os.path.join('pretrained', f'segformer_{url_hash}.pth')
            if not os.path.isfile(target_path):
                urllib.request.urlretrieve(pretrained_backbone_path, target_path)
            pretrained_backbone_path = target_path

        self.backbone = eval(backbone_name)(pretrained_backbone_path=pretrained_backbone_path)
        self.backbone_name = backbone_name
        self.head = SegFormerHead()
        self.align_corners = align_corners
        self.pretrained_backbone_path = pretrained_backbone_path

    def forward(self, x, apply_activation=True):
        """
        Calculate the output for the input tensor
        Args:
            x (torch.Tensor): input tensor
            apply_activation (bool): whether to apply the activation function to the output of the network
                                     (default: True)
        """
        B, C, H, W = x.shape
        feats = self.backbone(x)
        head_out = self.head(feats)
        _, _, H_H, H_W = head_out.shape
        if H_H < H or H_W < W:
            head_out = resize(head_out, (H, W), mode='bilinear', align_corners=self.align_corners)
        if apply_activation:
            head_out = F.softmax(head_out)
        return head_out
