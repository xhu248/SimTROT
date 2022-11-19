# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from models.nnFormer import nnFormer
from models.nnFormer import Encoder as nnEncoder
from models.byol import BYOL
from models.simmim import SimMIM
from models.unetr_pretrain import PretrainVisionTransformer, PretrainVisionTransformerEncoder
from models.unetr import UNETR

from utils.utils import RandomMaskingGenerator, load_checkpoint
from utils.tumor_utils import get_loader, patchify_augmentation
from trainer_pretrain import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--dataset', default='Task03_tumor', type=str, help='dataset directory')
parser.add_argument("--json_file", default="jsons/data_tumor.json", type=str, help="directory to the dataset json file")
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--num_steps', default=2500, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1.5e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--val_every', default=5, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--model', default='nnformer', type=str, help='model name')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=96, type=int, help='feature size dimention')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=4, type=int, help='number of output channels')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=2.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')

parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--patch_size', default=16, type=int, help='dimension of patch ')
parser.add_argument('--mask_scale', default=1, type=int, help='determine the mask size')
parser.add_argument('--mask_patch_size', default=8 , type=int, help='determine the mask size')
parser.add_argument('--pretrain_mode', default='mae', type=str, help='choose the pretraining mode')
parser.add_argument('--do_patch', action='store_true', help='whether calculate feature in patch-wise or do the average')
parser.add_argument('--do_rotation', action='store_true', help='whether calculate feature in patch-wise or do the average')
parser.add_argument('--add_contrast_mask', action='store_true', help='add noisy mask when doing patch contrastive')
parser.add_argument('--use_hybrid', action='store_true', help='choose whether to add a conv encoder before the vit')
parser.add_argument('--atten_weight', default=5.0, type=float, help='choose whether to add a conv encoder before the vit')


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)


def load_checkpoint(model, args):
    model_path = "../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task002_Synapse" + \
                   "/nnFormerTrainerV2_nnformer_synapse__nnFormerPlansv2.1/fold_0/model_best.model"
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = dict()
    model_keys = model.state_dict().keys()
    load_keys = []
    missing_keys = []

    if args.model == "nnformer":
        # get useful items in state_dict
        for key in state_dict.keys():
            if "model_down" in key:
                new_key = key.replace("model_down.", "")
                new_state_dict[new_key] = state_dict[key]
                if new_key in model_keys:
                    load_keys.append(new_key)
            else:
                new_state_dict[key] = state_dict[key]
                if key in model_keys:
                    load_keys.append(key)
    else:
        raise NotImplementedError("the model is currently not supported for feature inference")

    for key in model_keys:
        if key not in new_state_dict.keys():
            missing_keys.append(key)

    print("Missing keys:", missing_keys)
    print("Load keys from", model_path)
    print("Loaded keys:", load_keys)
    model.load_state_dict(new_state_dict, strict=False)


def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    args.loader_mode = "train"

    loader = get_loader(args)
    img_size = (args.roi_x, args.roi_y, args.roi_z)

    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.num_steps)
    inf_size = [args.roi_x, args.roi_y, args.roi_x]
    pretrained_dir = args.pretrained_dir

    if args.model == "nnformer":
        if 'contrast' in args.pretrain_mode or 'rotation' in args.pretrain_mode:
            model = nnEncoder(pretrain_img_size=img_size,
                              window_size=[4, 4, 8, 4],
                              embed_dim=args.feature_size,
                              patch_size=[4, 4, 4],
                              depths=[2, 2, 2, 2],
                              num_heads=[3, 6, 12, 24],
                              in_chans=args.in_channels)
        elif 'byol' in args.pretrain_mode:
            encoder = nnEncoder(pretrain_img_size=img_size,
                                window_size=[4, 4, 8, 4],
                                embed_dim=args.feature_size,
                                patch_size=[4, 4, 4],
                                depths=[2, 2, 2, 2],
                                num_heads=[3, 6, 12, 24],
                                in_chans=args.in_channels)
            model = BYOL(
                encoder,
                projection_hidden_size=encoder.embed_dim * (2 ** (encoder.num_layers - 1)),
                do_patch=args.do_patch
            )

        elif 'simmim' in args.pretrain_mode:
            args.add_contrast_mask = True
            encoder = nnEncoder(pretrain_img_size=img_size,
                                window_size=[4, 4, 8, 4],
                                embed_dim=args.feature_size,
                                patch_size=[4, 4, 4],
                                depths=[2, 2, 2, 2],
                                num_heads=[3, 6, 12, 24],
                                in_chans=args.in_channels)
            model = SimMIM(encoder=encoder, encoder_stride=32, out_channels=32**3, mask_patch_size=args.mask_patch_size)
        elif 'moco' in args.pretrain_mode:
            encoder = nnEncoder(pretrain_img_size=img_size,
                                window_size=[4, 4, 8, 4],
                                embed_dim=args.feature_size,
                                patch_size=[4, 4, 4],
                                depths=[2, 2, 2, 2],
                                num_heads=[6, 12, 24, 48],
                                in_chans=args.in_channels)
            encoder_k = nnEncoder(pretrain_img_size=img_size,
                                window_size=[4, 4, 8, 4],
                                embed_dim=args.feature_size,
                                patch_size=[4, 4, 4],
                                depths=[2, 2, 2, 2],
                                num_heads=[6, 12, 24, 48],
                                in_chans=args.in_channels)
            # moco requires a large number of negatives, which is not suitable for this dataset
            # model = MoBY(
            #     cfg=config,
            #     encoder=encoder,
            #     encoder_k=encoder_k,
            #     contrast_momentum=0.99,
            #     contrast_temperature=0.2,
            #     contrast_num_negative=4096,
            #     proj_num_layers=2,
            #     pred_num_layers=2,
            # )
    elif args.model == "unetr":
        if 'contrast' in args.pretrain_mode:
            model = PretrainVisionTransformerEncoder(
            img_size=img_size,
            in_chans=args.in_channels,
            embed_dim=768,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=torch.nn.LayerNorm,
            init_values=0.,
            )
    elif args.model == "unetr_pretrain":
        model = PretrainVisionTransformer(
            in_chans=args.in_channels,
            img_size=[128, 128, 128],
            encoder_embed_dim=768,
            decoder_num_classes=16*16*16*args.in_channels
        )
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))

    window_size = 128*128*128 / (args.patch_size**3)
    masked_position_generator = RandomMaskingGenerator(window_size, args.mask_ratio)

    dice_loss = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=args.smooth_nr,
                           smooth_dr=args.smooth_dr)
    post_label = AsDiscrete(to_onehot=True,
                            n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=True,
                           n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)

    best_acc = 0
    start_epoch = 0

    if args.resume_ckpt:
        load_checkpoint(model, args)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)
    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight,
                                      betas=(0.9, 0.95))
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.num_steps)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.num_steps)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    run_training(model=model,
                 train_loader=loader,
                 optimizer=optimizer,
                 mask_generator=masked_position_generator,
                 args=args,
                 model_inferer=model_inferer,
                 scheduler=scheduler,
                 start_epoch=start_epoch,
                 patch_augmentation=patchify_augmentation)
    return


if __name__ == '__main__':
    main()
