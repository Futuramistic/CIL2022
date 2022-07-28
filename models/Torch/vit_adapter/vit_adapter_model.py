import torch.nn as nn
import torch.nn.functional as F

from .beit_adapter import BEiTAdapter
from .msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from .sine_pos_encoding import SinePositionalEncoding
from .detr_transformer_decoder import DetrTransformerDecoder

from .mask2former_head import Mask2FormerHead
from .decode_head import resize


class ViTAdapter(nn.Module):
    def __init__(self):
        super(ViTAdapter, self).__init__()
        self.backbone = BEiTAdapter(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-6, drop_path_rate=0.3, conv_inplane=64, n_points=4, deform_num_heads=16, cffn_ratio=0.25, deform_ratio=0.5, interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]])

        self.head = Mask2FormerHead(in_channels=[1024, 1024, 1024, 1024],
                                    feat_channels=1024, out_channels=1024,
                                    num_queries=100,
                                     pixel_decoder=dict(
                                    type='MSDeformAttnPixelDecoder',
                                    num_outs=3,
                                    norm_cfg=dict(type='GN', num_groups=32),
                                    act_cfg=dict(type='ReLU'),
                                    encoder=dict(
                                        type='DetrTransformerEncoder',
                                        num_layers=6,
                                        transformerlayers=dict(
                                            type='BaseTransformerLayer',
                                            attn_cfgs=dict(
                                                type='MultiScaleDeformableAttention',
                                                embed_dims=1024,
                                                num_heads=32,
                                                num_levels=3,
                                                num_points=4,
                                                im2col_step=64,
                                                dropout=0.0,
                                                batch_first=False,
                                                norm_cfg=None,
                                                init_cfg=None),
                                            ffn_cfgs=dict(
                                                type='FFN',
                                                embed_dims=1024,
                                                feedforward_channels=4096,
                                                num_fcs=2,
                                                ffn_drop=0.0,
                                                act_cfg=dict(type='ReLU', inplace=True)),
                                            operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                                        init_cfg=None),
                                    positional_encoding=dict(
                                        type='SinePositionalEncoding', num_feats=512, normalize=True),
                                    init_cfg=None),
                                    positional_encoding=dict(type='SinePositionalEncoding', num_feats=512, normalize=True),
                                    transformer_decoder=dict(type='DetrTransformerDecoder',
                                                             return_intermediate=True,
                                                             num_layers=9,
                                                             transformerlayers=dict(
                                                                 type='DetrTransformerDecoderLayer',
                                                                 attn_cfgs=dict(
                                                                     type='MultiheadAttention',
                                                                     embed_dims=1024,
                                                                     num_heads=32,
                                                                     attn_drop=0.0,
                                                                     proj_drop=0.0,
                                                                     dropout_layer=None,
                                                                     batch_first=False),
                                                                 ffn_cfgs=dict(
                                                                     embed_dims=1024,
                                                                     feedforward_channels=4096,
                                                                     num_fcs=2,
                                                                     act_cfg=dict(type='ReLU', inplace=True),
                                                                     ffn_drop=0.0,
                                                                     dropout_layer=None,
                                                                     add_identity=True),
                                                                 feedforward_channels=4096,
                                                                 operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                                                                 'ffn', 'norm')),
                                                             init_cfg=None),
                                    in_index=[0, 1, 2, 3],
                                    loss_mask=dict(type='CrossEntropyLoss',
                                                   use_sigmoid=True,
                                                   reduction='mean'))

        # self.pixel_decoder = MSDeformAttnPixelDecoder()
        # self.positional_enc = SinePositionalEncoding(num_feats=512, normalize=True)
        # self.transformer_decoder = DetrTransformerDecoder(num_layers=9,
        #                                                   transformerlayers=dict(
        #                                                       type='DetrTransformerDecoderLayer',
        #                                                       attn_cfgs=dict(
        #                                                           type='MultiheadAttention',
        #                                                           embed_dims=1024,
        #                                                           num_heads=32,
        #                                                           attn_drop=0.0,
        #                                                           proj_drop=0.0,
        #                                                           dropout_layer=None,
        #                                                           batch_first=False),
        #                                                       ffn_cfgs=dict(
        #                                                           embed_dims=1024,
        #                                                           feedforward_channels=4096,
        #                                                           num_fcs=2,
        #                                                           act_cfg=dict(type='ReLU', inplace=True),
        #                                                           ffn_drop=0.0,
        #                                                           dropout_layer=None,
        #                                                           add_identity=True),
        #                                                       feedforward_channels=4096,
        #                                                       operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
        #                                                                        'ffn', 'norm')))
        
        # # "decode_head"
        # self.head = Mask2FormerHead(in_channels=[1024, 1024, 1024, 1024],
        #                             feat_channels=1024, out_channels=1024,
        #                             num_queries=100, pixel_decoder=self.pixel_decoder,
        #                             positional_encoding=self.positional_enc,
        #                             transformer_decoder=self.transformer_decoder,
        #                             in_index=[0, 1, 2, 3],
        #                             loss_mask=dict(type='CrossEntropyLoss',
        #                                            use_sigmoid=True,
        #                                            reduction='mean'))

    def forward(self, x, apply_activation=True):
        B, C, H, W = x.shape
        # upscale 
        
        if H < 512 or W < 512:
            x = resize(x, (512, 512), mode='bilinear', align_corners=False)

        # x = resize(x, (256, 256), mode='bilinear', align_corners=False)

        feats = self.backbone(x)
        head_out = self.head(feats, [{} for _ in range(len(x))])[-1][-1]
        _, _, H_H, H_W = head_out.shape
        if H_H < H or H_W < W:
            # see https://github.com/NVlabs/SegFormer/blob/08c5998dfc2c839c3be533e01fce9c681c6c224a/mmseg/models/segmentors/encoder_decoder.py
            head_out = resize(head_out, (H, W), mode='bilinear', align_corners=False)

        if apply_activation:
            head_out = F.sigmoid(head_out)
        return head_out
