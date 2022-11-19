
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.nnFormer import Encoder as nnFormerEncoder


# class VisionTransformerForSimMIM(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         assert self.num_classes == 0
#
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         self._trunc_normal_(self.mask_token, std=.02)
#
#     def _trunc_normal_(self, tensor, mean=0., std=1.):
#         trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
#
#     def forward(self, x, mask):
#         x = self.patch_embed(x)
#
#         assert mask is not None
#         B, L, _ = x.shape
#
#         mask_token = self.mask_token.expand(B, L, -1)
#         w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
#         x = x * (1 - w) + mask_token * w
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         if self.pos_embed is not None:
#             x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
#         for blk in self.blocks:
#             x = blk(x, rel_pos_bias=rel_pos_bias)
#         x = self.norm(x)
#
#         x = x[:, 1:]
#         B, L, C = x.shape
#         H = W = int(L ** 0.5)
#         x = x.permute(0, 2, 1).reshape(B, C, H, W)
#         return x
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels * 2 // self.scale ** 3

        out_depth = in_depth * self.scale // 2
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale // 2, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, out_channels, mask_patch_size):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.encoder.num_features[-1],
                out_channels=out_channels, kernel_size=1),
            PixelShuffle3d(self.encoder_stride),
        )

        self.in_chans = 1
        self.patch_size = mask_patch_size

    def forward(self, x, x_mask, mask):

        z = self.encoder(x_mask)
        if isinstance(z, list):
            z = z[-1]
        x_rec = self.decoder(z)

        patch_dim = x.shape[-1] // self.patch_size
        mask = mask.view(x.shape[0], patch_dim, patch_dim, patch_dim)
        mask = mask.repeat_interleave(self.patch_size // 2, 1).repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

#
# def build_simmim(config):
#     model_type = config.MODEL.TYPE
#     if model_type == 'swin':
#         encoder = SwinTransformerForSimMIM(
#             img_size=config.DATA.IMG_SIZE,
#             patch_size=config.MODEL.SWIN.PATCH_SIZE,
#             in_chans=config.MODEL.SWIN.IN_CHANS,
#             num_classes=0,
#             embed_dim=config.MODEL.SWIN.EMBED_DIM,
#             depths=config.MODEL.SWIN.DEPTHS,
#             num_heads=config.MODEL.SWIN.NUM_HEADS,
#             window_size=config.MODEL.SWIN.WINDOW_SIZE,
#             mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#             qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#             qk_scale=config.MODEL.SWIN.QK_SCALE,
#             drop_rate=config.MODEL.DROP_RATE,
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             ape=config.MODEL.SWIN.APE,
#             patch_norm=config.MODEL.SWIN.PATCH_NORM,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT)
#         encoder_stride = 32
#     else:
#         raise NotImplementedError(f"Unknown pre-train model: {model_type}")
#
#     model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
#
#     return model
