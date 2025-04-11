import os
from torch.hub import load_state_dict_from_url

# URL для скачивания весов предобученных моделей
urls = [
    # ResNets (-50, -101, -152)
    'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'https://download.pytorch.org/models/resnet152-394f9c45.pth',

    # EfficientNets (S, M, L)
    'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
    'https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth',
]

# Путь для сохранения весов
model_dir = os.path.dirname(__file__)

# Загрузка весов
for url in urls:
    state_dict = load_state_dict_from_url(url, model_dir=model_dir)