#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        r) train="true";;
        p) predict="true";;
        
    esac
done
echo $name	


#if ${train}
#then
#
#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer/nnformer/
#	CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2_nnformer_tumor 3 0
#fi

pretrain_mode=prot_contrast_w5_m0.85
if ${predict}
then

#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/
#	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name}_${pretrain_mode} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name} --pretrain_mode ${pretrain_mode}
#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
#	python nnformer/inference_synapse.py ${name}_${pretrain_mode}

#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/
#	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name}
#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
#	python nnformer/inference_synapse.py ${name}

#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/
#	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name}
#  cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
#	python nnformer/inference_tumor.py ${name}

# with pretrain weight

	pretrain_mode=prot_byol_m0.75_sgd
	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name}_${pretrain_mode} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name} --pretrain_mode ${pretrain_mode}
  cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
	python nnformer/inference_tumor.py ${name}_${pretrain_mode}


#	pretrain_mode=prot_contrast_w10_m0.75_sgd
#	cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/
#	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name}_${pretrain_mode} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr segmentTrainerV2_${name} --pretrain_mode ${pretrain_mode}
#  cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/nnFormer
#	python nnformer/inference_tumor.py ${name}_${pretrain_mode}

fi


## model save dir:
#
## dice results dir:
# cd /afs/crc.nd.edu/user/x/xhu7/Private/research_project/2022/vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/inferTs



