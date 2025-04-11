import torch
from typing import Literal

from models.vgg import ClassifierVGG
from models.resnet import ResNetClassifier, FaceEmbedder
from models.efficientnet import EfficientNetV2Classifier

###################
###################
###################
def vgg16(num_classes: int, path_to_weights: str = None):
    model = ClassifierVGG(num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def resnet50(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('classic', 50, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def resnet101(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('classic', 101, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def resnet152(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('classic', 152, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def ir50(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('improved', 50, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def ir101(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('improved', 101, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def ir152(num_classes: int, path_to_weights: str = None):
    model = ResNetClassifier('improved', 152, num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def face_embedder(model_size: Literal[50, 101, 152], embedding_size: int, path_to_weights: str = None):
    model = FaceEmbedder(model_size, embedding_size)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def efficientnet_v2_s(num_classes: int, path_to_weights: str = None):
    model = EfficientNetV2Classifier('s', num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def efficientnet_v2_m(num_classes: int, path_to_weights: str = None):
    model = EfficientNetV2Classifier('m', num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model

###################
###################
###################
def efficientnet_v2_l(num_classes: int, path_to_weights: str = None):
    model = EfficientNetV2Classifier('l', num_classes)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights))

    return model
