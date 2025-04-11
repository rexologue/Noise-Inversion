import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision
from models.resnet import ResNet
from models.efficientnet import EfficientNetV2

MODEL_NAME = ['resnet50', 'resnet101', 'resnet152', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'][3]
CHECKPOINT_TO_COMPARE = 'loaded_models/efficientnet_v2_s.pth'
SAVE_PATH = os.path.dirname(__file__)

ckp = torch.load(CHECKPOINT_TO_COMPARE, weights_only=False)

if MODEL_NAME.startswith('resnet'):
    model = ResNet('classic', int(MODEL_NAME.split('resnet')[1]))
    ckp_model = torchvision.models.resnet50(weights=None)

elif MODEL_NAME.startswith('efficientnet_v2'):
    model = EfficientNetV2(MODEL_NAME.split('efficientnet_v2_')[1])
    ckp_model = torchvision.models.efficientnet_v2_s(weights=None)


with open(os.path.join(SAVE_PATH, 'compare_arch.txt'), 'w', encoding='utf8') as f:
    f.write(str(model))
    f.write('\n\n' + '-'*80 + '\n' + '-'*80 + '\n\n')
    f.write(str(ckp_model))


def is_bn_key(key: str):
    after_last_dot = key.split('.')[-1]

    return after_last_dot in ['running_mean', 'running_var', 'num_batches_tracked']


with open(os.path.join(SAVE_PATH, 'compare_weights.txt'), 'w', encoding='utf8') as f:
    cur_dict = model.state_dict()
    
    cur_keys = list(cur_dict.keys())
    old_keys = [i for i in list(ckp.keys())] # if not is_bn_key(i)]
    
    keys_len = 100
    
    all_len = max(len(cur_keys), len(old_keys))
    
    for i in range(all_len):
        f.write(f"{i}: ")
        if i < len(cur_keys):
            v = f"{cur_keys[i]}: {cur_dict[cur_keys[i]].shape}"
            f.write(v + (keys_len - len(v))*" ")
        if i < len(old_keys):
            f.write(f"{old_keys[i]}: {ckp[old_keys[i]].shape}")
            
        f.write("\n")