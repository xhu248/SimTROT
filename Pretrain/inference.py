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

import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
# from utils.data_utils import get_loader
from utils.synapse_utils import get_loader
from utils.ops import aug_rand, rot_rand

from utils.data_utils import get_augmentation, patchify_augmentation


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def inference(data_loader, log_dir):
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                val_inputs = batch["image"].cuda()
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    xout_1 = model.forward_feature(x1_augment)
                    xout_2 = model.forward_feature(x2_augment)
                    imgs = torch.cat([x1, x2], dim=0)

                    print(len(xout_1))

                if step == 1:
                    break
        imgs = imgs.cpu().numpy()
        print("saving imgs to %s..." % log_dir)
        np.save(log_dir + "/imgs.npy", imgs[:, ::8])

        print("saveing features to %s ..." % log_dir)
        for i in range(len(xout_1)):
            xdata = torch.cat([xout_1[i], xout_2[i]], dim=0)
            xdata = xdata.cpu().numpy()
            np.save(log_dir + "/feature_layer" + str(i) + ".npy", xdata[::4])

    def inference_contrast(data_loader, log_dir):
        model.eval()
        aug_list = get_augmentation(args)
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # batch1 = aug_list(batch_data)
                # batch2 = aug_list(batch_data)
                #
                # x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)

                aug_batch, mask_index = patchify_augmentation(args, batch_data)

                x1, x2 = batch_data['image'].cuda(args.rank), aug_batch['image'].cuda(args.rank)
                with autocast(enabled=args.amp):
                    z1, z2 = model(x1)[-1], model(x2)[-1]
                    imgs = torch.cat([x1, x2], dim=0)

                if step == 1:
                    break
        imgs = imgs.cpu().numpy()
        print("saving imgs to %s..." % log_dir)
        np.save(log_dir + "/imgs.npy", imgs[:, ::8])

        print("saveing features to %s ..." % log_dir)
        xdata = torch.cat([z1, z2], dim=0).cpu().numpy()
        np.save(log_dir + "/feature_emb" +".npy", xdata[::4])

    def load_checkpoint():
        model_path = args.resume
        state_dict = torch.load(model_path)['state_dict']
        new_state_dict = dict()
        model_keys = model.encoder.state_dict().keys()
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
        model.encoder.load_state_dict(new_state_dict)

    ####################################### main part #############################################

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=192, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    # new added arguments
    parser.add_argument("--model", default="nnformer", type=str, help="optimization algorithm")
    parser.add_argument("--dataset", default="Task02_Synapse", type=str, help="optimization algorithm")
    parser.add_argument("--json_file", default="jsons/data_btcv.json", type=str, help="optimization algorithm")
    parser.add_argument("--test_mode", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--pretrain_mode", default="rcr", type=str, help="optimization algorithm")
    parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
    parser.add_argument('--workers', default=8, type=int, help='number of workers')

    # for data augmentation
    parser.add_argument('--RandFlipd_prob', default=0.0, type=float, help='RandFlipd aug probability')
    parser.add_argument('--RandRotate90d_prob', default=0.0, type=float, help='RandRotate90d aug probability')
    parser.add_argument('--RandScaleIntensityd_prob', default=0.0, type=float,
                        help='RandScaleIntensityd aug probability')
    parser.add_argument('--RandShiftIntensityd_prob', default=0.0, type=float,
                        help='RandShiftIntensityd aug probability')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--patch_size', default=16, type=int, help='dimension of patch ')
    parser.add_argument('--mask_scale', default=1, type=int, help='determine the mask size')
    parser.add_argument('--mask_patch_size', default=16, type=int, help='determine the mask size')
    parser.add_argument('--add_contrast_mask', action='store_true', help='add noisy mask when doing patch contrastive')

    args = parser.parse_args()
    logdir = "./runs/" + args.logdir + "/pcontrast_no_pretraining"
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)

    model = SSLHead(args)
    model.cuda()

    # args.resume = "../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task002_Synapse" + \
    #               "/nnFormerTrainerV2_nnformer_synapse__nnFormerPlansv2.1/fold_0/model_best.model"
    # args.resume = "./runs/results/nnformer_contrast/model_pretrain_contrast.pt"
    if args.resume is not None:
        load_checkpoint()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    train_loader, test_loader = get_loader(args)

    inference_contrast(train_loader, logdir)


if __name__ == "__main__":
    main()
