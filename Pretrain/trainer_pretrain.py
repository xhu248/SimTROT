import os
import time
import sys
import math
import shutil
import numpy as np
from einops import rearrange
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
import utils.utils as utils
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch
from utils.utils import get_augmentation, pos_rand, restore_feature
from models.ssl_head import MLP, PCHead
from utils.loss_functions import NTXentLoss
from typing import Iterable
import torch.nn as nn


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def proj_feat(x, hidden_size, feat_size):
    x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_mae_epoch(model,
                    loader,
                    optimizer,
                    mask_generator,
                    scaler,
                    epoch,
                    loss_func,
                    args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        bool_masked_pos = torch.from_numpy(mask_generator()).cuda(args.rank).to(torch.bool)

        # prepare labels
        images_patch = rearrange(data, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=args.patch_size,
                                 p2=args.patch_size, p3=args.patch_size)
        B, _, C = images_patch.shape
        labels = images_patch[:, bool_masked_pos].reshape(B, -1, C)
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data, bool_masked_pos)
            loss = loss_func(input=logits, target=labels)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg


def train_contrast_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                patch_augmentation,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    aug_list = get_augmentation(args)
    if args.model == "unetr":
        emb_dim = 768
    else:
        if args.distributed:
            emb_dim = model.module.embed_dim * (2 ** (model.module.num_layers - 1))
        else:
            emb_dim = model.embed_dim * (2 ** (model.num_layers - 1))
    mlp = MLP(input_dim=emb_dim).cuda(args.gpu)
    z1_list = []
    z2_list = []
    for idx, batch_data in enumerate(loader):

        for param in model.parameters(): param.grad = None
        for param in mlp.parameters(): param.grad = None

        if args.do_rotation:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)
            args.do_patch = True
            batch2, mask_index = patch_augmentation(args, batch2)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)
            x1, rots, flips = pos_rand(x1)
            with autocast(enabled=args.amp):
                if args.model == 'unetr':
                    z1, z2 = model(x1), model(x2)
                    # for tumor dataset
                    # z1 = proj_feat(z1, hidden_size=emb_dim, feat_size=(4, 8, 8))
                    # z2 = proj_feat(z2, hidden_size=emb_dim, feat_size=(4, 8, 8))
                    # for synapse dataset
                    z1 = proj_feat(z1, hidden_size=emb_dim, feat_size=(8, 8, 8))[:,:,::2]
                    z2 = proj_feat(z2, hidden_size=emb_dim, feat_size=(8, 8, 8))[:,:,::2]
                else:
                    z1, z2 = model(x1)[-1], model(x2)[-1]
                z1 = restore_feature(z1, rots, flips)
                z1, z2 = z1.reshape(-1, z1.shape[1]), z2.reshape(-1, z2.shape[1])
                f1, f2 = mlp(z1), mlp(z2)  # (B, 512)
                # f1, f2 = f1[mask_index == 1],  f2[mask_index == 1]
                loss = loss_func(f1, f2, True, args.atten_weight)

        elif args.do_patch:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)
            batch2, mask_index = patch_augmentation(args, batch2)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)

            with autocast(enabled=args.amp):
                if args.model == 'unetr':
                    z1, z2 = model(x1), model(x2)
                else:
                    z1, z2 = model(x1)[-1], model(x2)[-1]
                z1, z2 = z1.reshape(-1, z1.shape[1]), z2.reshape(-1, z2.shape[1])
                f1, f2 = mlp(z1), mlp(z2)  # (B, 512)
                # f1, f2 = f1[mask_index == 1],  f2[mask_index == 1]
                loss = loss_func(f1, f2)
        else:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)

            with autocast(enabled=args.amp):
                if args.model == 'unetr':
                    z1, z2 = model(x1), model(x2)
                else:
                    z1, z2 = model(x1)[-1], model(x2)[-1]
                z1, z2 = z1.mean(dim=(2, 3, 4)), z2.mean(dim=(2, 3, 4))
                f1, f2 = mlp(z1), mlp(z2)  # (B, 512)
                loss = loss_func(f1, f2)
        # data, target = data.cuda(args.rank), target.cuda(args.rank)
        # prepare labels

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()

        # save features when the loss get stable
        if epoch == 40:
            z1_list.append(z1.cpu().detach().numpy())
            z2_list.append(z2.cpu().detach().numpy())

            if len(z1_list) > 8:
                break


            # np.save(args.logdir + "/x1_e1.npy", x1)
            # np.save(args.logdir + "/x2_e1.npy", x2)

    # print("saveing features to %s ..." % args.logdir)
    # np.save(args.logdir + "/z1_list_e30.npy", np.asarray(z1_list))
    # np.save(args.logdir + "/z2_list_e30.npy", np.asarray(z2_list))
    #
    # if epoch == 40:
    #     exit()

    for param in model.parameters() : param.grad = None

    return run_loss.avg


def train_reconstruct_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                patch_augmentation,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']

        aug_batch, mask_index = patch_augmentation(args, batch_data)
        data = batch_data['image'].cuda(args.rank)
        aug_data = aug_batch['image'].cuda(args.rank)


        # bool_masked_pos = torch.from_numpy(mask_generator()).cuda(args.rank).to(torch.bool)

        # prepare labels
        data_patch = rearrange(data, 'b c (h p1) (w p2) (d p3) -> (b h w d) (p1 p2 p3 c)', p1=args.patch_size,
                                 p2=args.patch_size, p3=args.patch_size)
        B, C = data.shape[0], data_patch.shape[-1]
        if args.add_contrast_mask:
            labels = data_patch[mask_index == 1].reshape(-1, C)
        else:
            labels = data_patch.reshape(B, -1, C)

        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            logits = model(aug_data)
            logits_patch = rearrange(logits, 'b c (h p1) (w p2) (d p3) -> (b h w d) (p1 p2 p3 c)', p1=args.patch_size,
                                 p2=args.patch_size, p3=args.patch_size)
            if args.add_contrast_mask:
                logits_patch = logits_patch[mask_index == 1].reshape(-1, C)
            else:
                logits_patch = logits_patch.reshape(B, -1, C)
            loss = loss_func(input=logits_patch, target=labels)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg


def train_dvae_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):

        x = batch_data['image'].cuda(args.rank)
        # data, target = data.cuda(args.rank), target.cuda(args.rank)
        # prepare labels

        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            loss, _ = model(x)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

"""
def train_beit_epoch(model, d_vae, loader, optimizer, mask_generator, scaler, epoch, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):

        x = batch_data['image'].cuda(args.rank)
        bool_masked_pos = torch.from_numpy(mask_generator()).cuda(args.rank).to(torch.bool)

        # data, target = data.cuda(args.rank), target.cuda(args.rank)
        # prepare labels

        with torch.no_grad():
            logits, input_ids = d_vae.get_codebook_indices(x)
            _, out = d_vae(x)
            # bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            # labels = input_ids[bool_masked_pos]

            logits = logits.cpu().numpy()
            out = out.cpu().numpy()
            img = x.cpu().numpy()
            np.save(os.path.join(args.logdir, "logits.npy"), logits)
            np.save(os.path.join(args.logdir, "rect.npy"), out)
            np.save(os.path.join(args.logdir, "img.npy"), img)

        print(input_ids.shape, out.shape)
        exit()

        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            loss, _ = model(x)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg
"""


def train_beit_epoch(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast():
            outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
            loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()

        metric_logger.update(mlm_acc=mlm_acc)
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_byol_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                patch_augmentation,
                args):

    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    aug_list = get_augmentation(args)

    for idx, batch_data in enumerate(loader):
        if args.distributed:
            model.module.update_moving_average()
        else:
            model.update_moving_average()

        if args.do_rotation:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)
            args.do_patch = True
            batch2, mask_index = patch_augmentation(args, batch2)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)
            if x1.size(0) <= 1:
                continue
            x1, rots, flips = pos_rand(x1)
        elif args.do_patch:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)
            batch2, mask_index = patch_augmentation(args, batch2)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)
            if x1.size(0) <= 1:
                continue
        else:
            batch1 = aug_list(batch_data)
            batch2 = aug_list(batch_data)

            x1, x2 = batch1['image'].cuda(args.rank), batch2['image'].cuda(args.rank)
        # data, target = data.cuda(args.rank), target.cuda(args.rank)
        # prepare labels
            if x1.size(0) <= 1:
                continue
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            if args.do_rotation:
                loss, z1, z2 = model(x1, x2, rots=rots, flips=flips)
            else:
                loss, z1, z2 = model(x1, x2)
            if loss != loss:
                exit("Loss is NaN")
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()

    if epoch == 150:
        z1 = z1.cpu().detach().numpy()
        z2 = z2.cpu().detach().numpy()

        print("saveing features to %s ..." % args.logdir)
        np.save(args.logdir + "/z1_e1.npy", z1)
        np.save(args.logdir + "/z2_e1.npy", z2)

        # np.save(args.logdir + "/x1_e1.npy", x1)
        # np.save(args.logdir + "/x2_e1.npy", x2)


    for param in model.parameters() : param.grad = None

    return run_loss.avg


def train_simmim_epoch(model,
                    loader,
                    optimizer,
                    mask_generator,
                    scaler,
                    epoch,
                    loss_func,
                    patch_augmentation,
                    args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        batch2, mask_index = patch_augmentation(args, batch_data)
        mask_data = batch2['image'].cuda(args.rank)

        mask_index = torch.from_numpy(mask_index).cuda(args.rank)
        # bool_masked_pos = torch.from_numpy(mask_generator()).cuda(args.rank).to(torch.bool)

        # prepare labels
        images_patch = rearrange(data, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=args.patch_size,
                                 p2=args.patch_size, p3=args.patch_size)
        B, _, C = images_patch.shape
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            loss = model(data, mask_data, mask_index)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.num_steps, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg


def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(model,
                 train_loader,
                 optimizer,
                 mask_generator,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None,
                 d_vae=None,
                 patch_augmentation=None,
                 ):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.
    for epoch in range(start_epoch, args.num_steps):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        if args.pretrain_mode == "mae":
            train_loss = train_mae_epoch(model,
                                         train_loader,
                                         optimizer,
                                         mask_generator,
                                         scaler=scaler,
                                         epoch=epoch,
                                         loss_func=torch.nn.MSELoss(),
                                         args=args)
        elif "contrast" in args.pretrain_mode :
            train_loss = train_contrast_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                epoch=epoch,
                loss_func=NTXentLoss(temperature=0.5),
                args=args,
                patch_augmentation=patch_augmentation
            )

        elif args.pretrain_mode == "vae":
            train_loss = train_dvae_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                epoch=epoch,
                args=args
            )

        elif args.pretrain_mode == "reconstruct":
            train_loss = train_reconstruct_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                epoch=epoch,
                args=args,
                loss_func=torch.nn.MSELoss(),
            )

        elif "byol" in args.pretrain_mode:
            train_loss = train_byol_epoch(
                model,
                train_loader,
                optimizer,
                scaler=scaler,
                epoch=epoch,
                args=args,
                patch_augmentation=patch_augmentation
            )

        elif args.pretrain_mode == "beit":
            train_loss = train_beit_epoch(model,
                                     d_vae,
                                     train_loader,
                                     optimizer,
                                     mask_generator,
                                     scaler=scaler,
                                     epoch=epoch,
                                     args=args)
        elif args.pretrain_mode == "simmim":
            train_loss = train_simmim_epoch(model,
                                         train_loader,
                                         optimizer,
                                         mask_generator,
                                         scaler=scaler,
                                         epoch=epoch,
                                         loss_func=torch.nn.MSELoss(),
                                         args=args,
                                         patch_augmentation=patch_augmentation)
        else:
            raise NotImplementedError("The pretraining mode %s is not supported yet!" % args.pretrain_mode)
        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.num_steps - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False

        if (epoch + 1) % args.val_every == 0:
            save_checkpoint(model,
                            epoch,
                            args,
                            filename='model_pretrain_%s.pt' % args.pretrain_mode)

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !)')


