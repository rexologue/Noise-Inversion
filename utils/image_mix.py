import torch
import numpy as np

def mixup(images: torch.Tensor, labels: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    """
    Реализация MixUp-аугментации.

    Алгоритм:
    1. Для батча из B изображений формируем случайные значения λ, используя Beta-распределение.
    (Например, Beta(0.4, 0.4)).
    2. Случайно перемешиваем (permute) индексы в батче, чтобы создать пары (image, image[idx]).
    3. Линейно смешиваем изображения:
        images_mix = λ * images + (1 - λ) * shuffled_images
    Аналогично смешиваем one-hot представления меток:
        labels_mix = λ * one_hot(labels) + (1 - λ) * one_hot(shuffled_labels)

    Args:
        images (torch.Tensor): Батч изображений формы (B, C, H, W).
        labels (torch.Tensor): Батч меток формы (B,) с целочисленными индексами классов.
        device (str): Девайс, на котором выполняются вычисления ('cuda' или 'cpu').

    Returns:
        torch.Tensor, torch.Tensor: преобразованные изображения изображения и метки с учётом MixUp-аугментации.
    """
    # Sample MixUp parameter (lambda)
    k = torch.from_numpy(np.random.beta(0.4, 0.4, size=images.size(0)).astype(np.float32)).to(device) 
    # Sample random permutation to randomize pairs that will be MixUped
    idx = torch.randperm(images.size(0)).to(device)

    # Broadcast to (B, C, H, W)
    k = k.view(-1, 1, 1, 1) 

    # Suffle images and labels
    shuffled_images = images[idx, ...]
    shuffled_labels = labels[idx]

    # Apply MixUp for images
    images_mix = k * images + (1 - k) * shuffled_images
    
    # Enocde labels to one-hot format
    ohes = torch.nn.functional.one_hot(labels, num_classes).float()
    shuffled_ohes = torch.nn.functional.one_hot(shuffled_labels, num_classes).float()

    # Broadcast to (B, num_classes)
    k = k.view(-1, 1)
    # Apply MixUp for labels
    labels_mix = k * ohes + (1 - k) * shuffled_ohes
    
    return images_mix, labels_mix


def cutmix(images: torch.Tensor, labels: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    """
    Реализует CutMix-аугментацию для батча изображений с формой (B, C, H, W).

    Чтобы понять, какие пиксели в каждом изображении нужно «вырезать» и заменить,
    мы создаём два вспомогательных тензора: `grid_y` (вертикальные координаты)
    и `grid_x` (горизонтальные координаты).

    - `grid_y[b, i, j] = i`: номер строки пикселя `i` для всех картинок в батче.
    Иными словами, это «вертикальная» координата пикселя.
    - `grid_x[b, i, j] = j`: номер столбца пикселя `j` для всех картинок в батче.
    Это «горизонтальная» координата пикселя.

    Пример координатных сеток (для одного элемента батча):
    grid_y:
        [0,   0,   0,   0,   ... ],
        [1,   1,   1,   1,   ... ],
        ...,
        [H-1, H-1, H-1, H-1, ... ]

    grid_x:
        [0,   1,   2,  ..., W-1],
        [0,   1,   2,  ..., W-1],
        ...,
        [0,   1,   2,  ..., W-1]

    Зачем это нужно? В операции CutMix мы «вырезаем» прямоугольный участок из
    одного изображения и заменяем им соответствующую часть другого. Сравнивая
    `grid_y` и `grid_x` с границами прямоугольника (`top`, `left`, высота и
    ширина вырезаемой области), мы определяем, попадает ли конкретный пиксель
    внутрь нужной зоны. Результатом является булева маска (True — «внутри»,
    False — «снаружи»). При помощи этой маски мы выборочно заменяем содержимое
    только в заданном участке.

    Args:
        images (torch.Tensor): Батч исходных изображений формы (B, C, H, W).
        labels (torch.Tensor): Вектор меток формы (B, ) для данного батча.
        device (str): Устройство, на котором выполняются вычисления ('cuda' или 'cpu').

    Returns:
        torch.Tensor, torch.Tensor: преобразованные изображения изображения и метки с учётом CutMix-аугментации.
    """
    B, C, H, W = images.shape

    # Sample random permutation to randomize pairs that will be CutMixed
    idx = torch.randperm(B).to(device)
    shuffled_images = images[idx, ...]
    shuffled_labels = labels[idx]

    # Sample CutMix parameter (lambda)
    k = torch.from_numpy(np.random.beta(0.4, 0.4, size=B).astype(np.float32)).to(device)

    # Define sizes of cropped areas
    new_hs = (k * H).to(dtype=torch.long)
    new_ws = (k * W).to(dtype=torch.long)

    # Make sure that new_hs and new_ws >= 1 and not higher H, and W respectivelly
    new_hs = torch.clamp(new_hs, min=1, max=H)
    new_ws = torch.clamp(new_ws, min=1, max=W)

    # Sample random coordinates for cropping (Left Top corner)
    rand_h_floats = torch.rand(new_hs.shape, device=device)
    rand_w_floats = torch.rand(new_ws.shape, device=device)

    top = (rand_h_floats * (H - new_hs + 1)).to(dtype=torch.long)
    left = (rand_w_floats * (W - new_ws + 1)).to(dtype=torch.long)

    # grid_y and grid_x:
    # We create 2D coordinates for each pixel in all the batches so we can compare “current (i, j)” to the rectangle boundaries.
    grid_y = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    grid_x = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

    mask = (grid_y >= top.view(-1, 1, 1)) & (grid_y < (top + new_hs).view(-1, 1, 1)) & \
        (grid_x >= left.view(-1, 1, 1)) & (grid_x < (left + new_ws).view(-1, 1, 1))

    # Expand mask to all channels
    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)

    # Apply CutMix
    images_mix = torch.where(mask, shuffled_images, images)
    
    # Enocde labels to one-hot format
    ohes = torch.nn.functional.one_hot(labels, num_classes).float()
    shuffled_ohes = torch.nn.functional.one_hot(shuffled_labels, num_classes).float()

    # Recalculate k to adjust real squares of croppings
    k = (new_hs * new_ws) / (H * W)
    k = k.view(-1, 1)
    
    # Apply CutMix for labels
    labels_mix = k * ohes + (1 - k) * shuffled_ohes
    
    return images_mix, labels_mix
