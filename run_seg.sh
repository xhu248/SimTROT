#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 8     # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N tmr_prot_contrast_unetr # Specify job name

module load conda
source activate nnFormer

PRETRAIN=prot_contrast_w5_m0.75_sgd
MODEL=nnformer

export OMP_NUM_THREADS=${NSLOTS}
#python nnformer/run/run_training.py 3d_fullres segmentTrainerV2_nnformer_tumor 3 0

# python nnformer/run/run_training.py 3d_fullres segmentTrainerV2_${MODEL}_tumor 3 0 \
# -pretrained_weights Task03_tumor/${MODEL}_${PRETRAIN}/model_pretrain_prot_contrast.pt -pretrain_mode ${PRETRAIN}

python nnformer/run/run_training.py 3d_fullres segmentTrainerV2_${MODEL}_tumor 3 0 \
-c -pretrain_mode ${PRETRAIN}

#python nnformer/run/run_training.py 3d_fullres segmentTrainerV2_${MODEL}_synapse 2 0 \
#-pretrained_weights ${MODEL}_${PRETRAIN}/model_pretrain_prot_contrast.pt -pretrain_mode ${PRETRAIN}

# python nnformer/run/run_training.py 3d_fullres nnFormerTrainerV2_${MODEL}_tumor 3 0 \

# --resume_ckpt --pretrained_dir='./runs/unetr_pretrain_mae' \
# --pretrained_model_name model_pretrain_contrast.pt

# bash train_inference.sh -c 2 -n nnformer_synapse -t 2
# CUDA_VISIBLE_DEVICES=1 python nnformer/run/run_training.py 3d_fullres segmentTrainerV2_swinunetr_tumor 3 0

