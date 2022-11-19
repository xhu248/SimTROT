#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnFormer architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")


def load_checkpoint(model, state_dict, pretrain_mode, model_name='nnformer'):
    if pretrain_mode != "mae":
        if hasattr(model, "model_down"):
            model = model.model_down
        elif hasattr(model, "vit"):
            model = model.vit
    new_state_dict = dict()
    model_keys = model.state_dict().keys()
    load_keys = []
    missing_keys = []

    if "byol" in pretrain_mode:
        for key in state_dict.keys():
            if "online_encoder" in key:
                new_key = key.replace('online_encoder.net.', '')
                new_state_dict[new_key] = state_dict[key]
                if new_key in model_keys:
                    load_keys.append(new_key)
    elif pretrain_mode == "rcr" or pretrain_mode == "simmim":
        for key in state_dict.keys():
            if "encoder" in key:
                new_key = key.replace('encoder.', '')
                new_state_dict[new_key] = state_dict[key]
                if new_key in model_keys:
                    load_keys.append(new_key)
    elif pretrain_mode == "mae":
        for key in state_dict.keys():
            if key.startswith('encoder.'):
                # pretrained with mae, only for unetr
                new_key = key.replace('encoder.', 'vit.')
                new_state_dict[new_key] = state_dict[key]
                if new_key in model_keys:
                    load_keys.append(new_key)
    else:
        for key in state_dict.keys():
            if key in model_keys:
                load_keys.append(key)

        new_state_dict = state_dict

    for key in model_keys:
        if key not in new_state_dict.keys():
            missing_keys.append(key)

    model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Load keys:", load_keys)
