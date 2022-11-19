import os
import json
import shutil

if __name__ == "__main__":
    with open("nnformer/dataset_json/tumor_dataset.json") as f:
        splits = json.load(f)
    test_file = splits["test"]

    # copy file from trainset folder to test folder
    base_dir = "../DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor"
    join = os.path.join
    for f in test_file:
        target_path = join(base_dir, f)
        # src_path = join(base_dir, f.replace("imagesTs", "imagesTr"))
        # # print(src_path, target_path)
        # if os.path.exists(src_path):
        #     print(src_path, target_path)
        #     print("moving file:", f)
        #     shutil.move(src_path, target_path)

        label = f.replace("imagesTs", "labelsTs")
        target_label_path = join(base_dir, label)
        src_label_path = join(base_dir, label.replace("labelsTs", "labelsTr"))

        print(src_label_path, target_label_path)
        if os.path.exists(src_label_path):
            print(src_label_path, target_label_path)
            print("coping file:", f)
            shutil.copy(src_label_path, target_label_path)

    # remove test_file in training folder and remove other unlabled data in test folder
