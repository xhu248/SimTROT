import torch
import os
import nibabel as nib
import numpy as np
import shutil

if __name__ == "__main__":
    # model_path = "./runs/results/nnformer_contrast/model_pretrain_contrast.pt"
    # model_path = "./runs/results/nnformer_contrast/model_pretrain_contrast.pt"
    #
    # state_dict = torch.load(model_path)['state_dict']
    # print(state_dict.keys())

    # data_dir = "../DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/imagesTr"
    data_dir = "../DATASET/nnFormer_preprocessed/Task003_tumor/nnFormerData_plans_v2.1_stage0"
    target_dir = "../DATASET/nnFormer_preprocessed/Task003_tumor/pretrain_data"
    join = os.path.join
    file_list = os.listdir(data_dir)

    for f in file_list:
        if ".npy" in f:
            img_path = join(data_dir, f)
            target_path = join(target_dir, f)
            shutil.copy(img_path, target_path)
            print(f)

        # img_path = join(data_dir, f)
        # img = nib.load(img_path)
        # data = img.get_data()
        #
        # print(f, data.shape)



