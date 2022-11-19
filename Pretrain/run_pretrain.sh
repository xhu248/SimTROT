#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 4    # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N unetr_protc_tmr_pretrain # Specify job name

module load conda
source activate nnFormer
export OMP_NUM_THREADS=${NSLOTS}
# run_crc
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main_rcr.py \
# --batch_size=1 --num_steps=1000 --lrdecay  --logdir=results/swinunetr_rcr --model swin
# python main_rcr.py --batch_size=1 --num_steps=1000 --lrdecay  --logdir=results/nnformer_rcr --model nnformer
# CUDA_VISIBLE_DEVICES=3 python main_rcr.py --batch_size=1 --num_steps=1000 --lrdecay  --logdir=results/swinunetr_rcr --model swin
# run SimCLR
#CUDA_VISIBLE_DEVICES=3 python main_pretraining.py --batch_size=1 --num_steps=1000 --pretrain_mode contrast \
# --model nnformer --logdir=results/nnformer_contrast --optim_lr 4e-4

# run mae
#CUDA_VISIBLE_DEVICES=0 python main_pretraining.py --batch_size=2 --num_steps=1000 --pretrain_mode mae \
# --model unetr_pretrain --logdir=results/unetr_mae --optim_lr 4e-4

# run BYOL
#python main_pretraining.py  --batch_size=1 --num_steps=1000 --pretrain_mode byol \
# --model nnformer --logdir=results/nnformer_byol_trsp --optim_lr 1e-4 --optim_name sgd


# run pcontrast
 # --do_patch --optim_name sgd
# CUDA_VISIBLE_DEVICES=2 python main_pretraining.py  --batch_size=1 --num_steps=500 --pretrain_mode pcontrast \
# --model nnformer --logdir=results/nnformer_pcontrast_no_trsp --optim_lr 1e-4 --optim_name sgd --do_patch  --add_contrast_mask
#
# CUDA_VISIBLE_DEVICES=2 python main_pretraining.py  --batch_size=1 --num_steps=500 --pretrain_mode prot_contrast \
# --model nnformer --logdir=results/nnformer_prot_contrast_atten_same_patch_w10 --optim_lr 1e-4 --optim_name sgd \
# --do_patch --do_rotation --add_contrast_mask --atten_weight 10
#python main_pretraining.py  --batch_size=1 --num_steps=500 --pretrain_mode prot_contrast \
# --model nnformer --logdir=results/nnformer_prot_contrast_w5_m0.85 --optim_lr 1e-4 --optim_name sgd --do_patch --do_rotation \
# --add_contrast_mask --mask_ratio 0.85
#
# python main_pretraining.py  --batch_size=1 --num_steps=500 --pretrain_mode prot_contrast \
# --model nnformer --logdir=results/nnformer_prot_contrast_w2_m0.75 --optim_lr 1e-4 --optim_name sgd --do_patch --do_rotation \
# --add_contrast_mask --atten_weight 2


######################### For Task02_tumor dataset #################################
#python main_tumor_pretraining.py  --batch_size=2 --num_steps=500 --pretrain_mode prot_contrast \
# --model nnformer --logdir=results/Task03_tumor/nnformer_prot_byol_m0.75_sgd --optim_lr 1e-4 --optim_name sgd \
# --do_patch  --do_rotation --add_contrast_mask

 python main_tumor_pretraining.py  --batch_size=1 --num_steps=500 --pretrain_mode prot_contrast \
 --model unetr --logdir=results/Task03_tumor/unetr_prot_contrast_m0.75 --optim_lr 1e-4 --optim_name sgd \
 --do_patch  --do_rotation --add_contrast_mask
# 2e-4
#
CUDA_VISIBLE_DEVICES=0 python main_tumor_rcr.py \
 --batch_size=2 --num_steps=2500 --lrdecay  --logdir=results/Task03_tumor/nnformer_rcr --model nnformer
#CUDA_VISIBLE_DEVICES=0 python main_tumor_pretraining.py  --batch_size=2 --num_steps=500 --pretrain_mode simmim \
# --model nnformer --logdir=results/Task03_tumor/nnformer_simmim --optim_lr 2e-4 --do_patch
