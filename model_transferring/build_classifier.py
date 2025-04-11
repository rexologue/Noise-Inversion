import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from models.resnet import ResNetClassifier
from models.efficientnet import EfficientNetV2Classifier

NUM_CLASSES = 101
MODEL_NAME = ['resnet50', 'resnet101', 'resnet152', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'][0]

SAVE_PATH = f'../weights/pretrained_{MODEL_NAME}_{NUM_CLASSES}.pth'
PRETRAINED_CHECKPOINT = f'../weights/imagenet_{MODEL_NAME}.pth'

model = torch.nn.Module()
pretrained_state = torch.load(PRETRAINED_CHECKPOINT, weights_only=False)

if MODEL_NAME.startswith('resnet'):
    model = ResNetClassifier('classic', int(MODEL_NAME.split('resnet')[1]), NUM_CLASSES)
    model.resnet.load_state_dict(pretrained_state)

elif MODEL_NAME.startswith('efficientnet_v2'):
    model = EfficientNetV2Classifier(MODEL_NAME.split('efficientnet_v2_')[1], NUM_CLASSES)
    model.efficientnet.load_state_dict(pretrained_state)


torch.save(model.state_dict(), SAVE_PATH)
