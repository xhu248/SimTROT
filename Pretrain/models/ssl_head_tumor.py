# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_unetr import SwinTransformer as SwinViT
from models.nnFormer import Encoder
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, args, upsample="deconv", dim=768):
        # dim=384 for nnFormer
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        if args.model == 'swin':
            self.encoder = SwinViT(
                in_chans=1,
                embed_dim=48,
                window_size=(7, 7, 7),
                patch_size=(4, 4, 4),
                depths=[2, 2, 2, 2],
                num_heads=[3, 6, 12, 24],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                norm_layer=torch.nn.LayerNorm,
                spatial_dims=3,
            )
        elif args.model == 'nnformer':
            self.encoder = Encoder(pretrain_img_size=(128, 128, 128),
                                   window_size=[4,4,8,4],
                                   embed_dim=args.feature_size,
                                   patch_size=[4,4,4],
                                   depths=[2,2,2,2],
                                   num_heads=[3,6,12,24],
                                   in_chans=args.in_channels)

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            # self.conv = nn.Sequential(
            #     nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            #     nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            #     nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            #     nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            #     nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # )
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 16, kernel_size=(4, 4, 4), stride=(4, 4, 4)),
                # nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, dim // 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 32, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=[2,1,1]),
            )
        else:
            print("no forward method, just for encoder inference")
            self.conv = None

    def forward(self, x):
        x_out = self.encoder(x.contiguous())[-1]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec

    def forward_feature(self, x):
        return self.encoder(x.contiguous())



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(input_dim, input_dim*2), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(input_dim*2, input_dim), nn.ReLU())

    def forward(self, x):
        # x input shape : [B*N, C]
        # x = x.mean(dim=(2,3,4))
        y = self.linear2(self.linear1(x))
        return F.normalize(y, dim=1)


class PCHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(input_dim, input_dim*2), nn.BatchNorm1d(input_dim*2),  nn.ReLU())
        self.linear2 = nn.Linear(input_dim*2, input_dim)

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        y = self.linear2(self.linear1(x))
        return F.normalize(y, dim=1)
