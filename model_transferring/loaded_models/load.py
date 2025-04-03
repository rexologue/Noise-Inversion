from torch.hub import load_state_dict_from_url

# URL для скачивания весов ResNet-50
urls = [
    'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'https://download.pytorch.org/models/resnet152-394f9c45.pth',
]

# Путь для сохранения весов
model_dir = '/home/super/mironov/mia/model_transferring/loaded_models'

# Загрузка весов
for url in urls:
    state_dict = load_state_dict_from_url(url, model_dir=model_dir)