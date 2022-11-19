#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 8          # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N infer_synp # Specify job name

module load conda
source activate nnFormer

task=2
pretrain_mode=contrast
name=nnformer_synapse

cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/
 nnFormer_predict -i imagesTs -o inferTs/${name}_${pretrain_mode} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name} --pretrain_mode ${pretrain_mode}
cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
python nnformer/inference_synapse.py ${name}_${pretrain_mode}

# -t 2 means synapse dataset, -t 3 means tumor dataset
# bash train_inference.sh -c 0 -n nnformer_tumor -t 3
# bash train_inference.sh -c 0 -n nnformer_synapse -t 2
