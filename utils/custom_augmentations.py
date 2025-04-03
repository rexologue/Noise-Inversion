import cv2
import torch
import numpy as np

from typing import Union, Tuple


class Compose:
    """
    Позволяет объединять несколько трансформаций в единый пайплайн.
    """
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
            
        return image



class Resize:
    """
    Ресайз изображения до заданного (height,width).
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        
        # Проверка корректности size
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError('Target size should be positive!')


    def __call__(self, image: np.ndarray) -> np.ndarray:
        h_orig, w_orig = image.shape[:2]
        w_new, h_new = self.size

        # Если уменьшаем по обеим осям — используем INTER_AREA, иначе INTER_CUBIC
        if w_new < w_orig and h_new < h_orig:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        return cv2.resize(image, (w_new, h_new), interpolation=interpolation)



class Normalize:
    """
    Нормировка пикселей в диапазоне [0..1] -> вычитание среднего, деление на std.
    mean и std принимаются как кортежи (mean_r, mean_g, mean_b).
    """
    def __init__(self, 
                 mean: Tuple[float, float, float], 
                 std: Tuple[float, float, float]):
        
        self.mean = np.array(mean, dtype=np.float32).reshape((1, 1, 3))
        self.std  = np.array(std,  dtype=np.float32).reshape((1, 1, 3))


    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Ожидается, что изображение уже в формате [0..1].
        return (image - self.mean) / self.std



class ToTensor:
    """
    Перевод изображения из numpy [H,W,C] в тензор Torch [C,H,W].
    """
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # image сейчас float32 или float64. Для PyTorch чаще используют float32
        tensor = torch.from_numpy(image).float()   # -> (H, W, C)
        tensor = tensor.permute(2, 0, 1)           # -> (C, H, W)
        return tensor



class ToFloat:
    """
    Приводит изображение к типу float32 и масштабирует в [0..1], если оно было в [0..255].
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Предполагается, что если в image есть значения > 1, это RGB с диапазоном 0..255
        if image.max() > 1.0:
            image /= 255.0
            
        return image
    
    
    
class RandomHorizontalFlip:
    """
    С вероятностью p отражает изображение горизонтально.
    """
    def __init__(self, p: float = 0.5):
        self.p = p


    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # np.fliplr() отражает по горизонтали
            image = np.fliplr(image)
            
        return image



class RandomVerticalFlip:
    """
    С вероятностью p отражает изображение вертикально.
    """
    def __init__(self, p: float = 0.5):
        self.p = p


    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # np.flipud() отражает по вертикали
            image = np.flipud(image)
            
        return image



class RandomRotate:
    """
    Случайный поворот изображения в диапазоне [-max_angle, max_angle] (в градусах).
    """
    def __init__(self, max_angle: float = 30.0):
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        
        # Матрица поворота вокруг центра
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        
        # warpAffine умеет обрабатывать цветные изображения (3 канала)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return rotated



class RandomCrop:
    """
    Случайная обрезка изображения до указанного масштаба (scale) от исходного.
    Масштаб может быть указан как в формате отрезка [p_min, p_max] и тогда значение будет выбрано из U[p_min, p_max].
    Иначе можно установить детерминированный масштаб.
    Опционально после обрезки изображение приводится к целевому размеру (output_size).
    """
    def __init__(self, 
                 scale: Union[Tuple[float, float], float],
                 output_size: Union[Tuple[int, int], None]):
        
        # Опционально устанавливаем параметры для выхода
        if output_size:
            self.size = output_size
            self.resizer = Resize(output_size)
        else:
            self.size = None
            
        self.scale = scale

        # Проверка корректности scale
        if isinstance(scale, tuple):
            if self.scale[0] > 1 or self.scale[1] > 1 or self.scale[0] < 0 or self.scale[1] < 0:
                raise ValueError('Crop scale should be in (0..1)!')
        else:
            if self.scale > 1 or self.scale < 0:
                raise ValueError('Crop scale should be in (0..1)!')
        

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Применяем случайное масштабирование
        if isinstance(self.scale, tuple):
            current_scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
        else:
            current_scale = self.scale
        
        h, w = image.shape[:2]
        new_h = max(1, int(current_scale * h))  # Убедимся, что new_h >= 1
        new_w = max(1, int(current_scale * w))  # Убедимся, что new_w >= 1
        
        # Вычисляем случайные координаты для обрезки
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Вырезаем соответствующую область
        image = image[top: top + new_h, left: left + new_w]
        
        # Приводим изображение к целевому размеру, если необходимо
        return image if self.size is None else self.resizer(image)



class CenterCrop:
    """
    Центрированная обрезка изображения до размера (height, width).
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size


    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = self.size

        # Вычисляем координаты верхнего левого угла кропа
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h, left: left + new_w]
        return image



class RandomBrightness:
    """
    Случайно изменяет яркость изображения в пределах заданного диапазона.

    Принцип работы:
    1. С вероятностью `p` изображение подвергается изменению яркости.
    2. Вычисляется коэффициент `alpha` как `1 + U[limit(0), limit(1)]`:
       - Если `alpha > 1`, изображение становится ярче (значения пикселей увеличиваются).
       - Если `alpha < 1`, изображение становится темнее (значения пикселей уменьшаются).
    3. Значения пикселей ограничиваются диапазоном [0, 1] для сохранения корректности.

    Параметры:
    - limit: Диапазон для случайного выбора коэффициента `alpha` (по умолчанию (-0.2, 0.2)).
    - p: Вероятность применения изменения яркости (по умолчанию 0.5).
    """
    def __init__(self, limit: Tuple[float, float] = (-0.2, 0.2), p: float = 0.5):
        self.limit = limit
        self.p = p
        
        if not isinstance(limit, Tuple) or len(limit) != 2:
            raise ValueError("limit must be a tuple of two values.")
        if limit[0] > limit[1]:
            raise ValueError("limit must to be in (min, max) format.")


    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # alpha > 1 -> ярче, alpha < 1 -> темнее
            alpha = 1.0 + np.random.uniform(self.limit[0], self.limit[1])
            image = np.clip(image * alpha, 0, 1)
            
        return image



class RandomContrast:
    """
    Случайно изменяет контраст изображения.

    Принцип работы:
    1. Вычисляется коэффициент `alpha` как `1 + U[limit(0), limit(1)]` 
    1. Изображение центрируется вокруг своего среднего значения (вычитается среднее).
       Т.о. темные пиксели имеют значения < 0, яркие > 0.
    2. Разброс значений пикселей изменяется умножением на коэффициент `alpha`:
       - Если `alpha > 1`, разброс увеличивается (контраст усиливается, 
         т.к. отрицательные значения становятся более отрицательными,
         а положительные - более положительными). 
       - Если `alpha < 1`, разброс уменьшается (контраст ослабляется, обратная логика).
    3. Изображение возвращается к исходному среднему значению (добавляется среднее).
    4. Значения пикселей ограничиваются диапазоном [0, 1] для сохранения корректности.

    Параметры:
    - limit: Диапазон для случайного выбора коэффициента `alpha` (по умолчанию (-0.2, 0.2)).
    - p: Вероятность применения изменения контраста (по умолчанию 0.5).
    """
    def __init__(self, limit: Tuple[float, float] = (-0.2, 0.2), p: float = 0.5):
        self.limit = limit
        self.p = p

        if not isinstance(limit, Tuple) or len(limit) != 2:
            raise ValueError("limit must be a tuple of two values.")
        if limit[0] > limit[1]:
            raise ValueError("limit must to be in (min, max) format.")


    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            alpha = 1.0 + np.random.uniform(self.limit[0], self.limit[1])
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = (image - mean) * alpha + mean
            image = np.clip(image, 0, 1)
            
        return image
    


class RandomSaturation:
    """
    Случайно изменяет насыщенность изображения.

    Принцип работы:
    1. С вероятностью `p` изображение подвергается изменению насыщенности.
    2. Изображение преобразуется в цветовое пространство HSV.
    3. Насыщенность (канал S) изменяется умножением на коэффициент `alpha`:
       - Если `alpha > 1`, насыщенность увеличивается.
       - Если `alpha < 1`, насыщенность уменьшается.
    4. Изображение возвращается в цветовое пространство RGB.
    5. Значения пикселей ограничиваются диапазоном [0, 1].

    Параметры:
    - limit: Диапазон для случайного выбора коэффициента `alpha` (по умолчанию (-0.2, 0.2)).
    - p: Вероятность применения изменения насыщенности (по умолчанию 0.5).
    """
    def __init__(self, limit: Tuple[float, float] = (-0.2, 0.2), p: float = 0.5):
        self.limit = limit
        self.p = p
        
        if not isinstance(limit, Tuple) or len(limit) != 2:
            raise ValueError("limit must be a tuple of two values.")
        if limit[0] > limit[1]:
            raise ValueError("limit must to be in (min, max) format.")


    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # Преобразуем изображение в HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Вычисляем коэффициент alpha
            alpha = 1.0 + np.random.uniform(self.limit[0], self.limit[1])
            
            # Изменяем насыщенность (канал S)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * alpha, 0, 1)
            
            # Возвращаем изображение в RGB
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return image
    
    
    
class ColorJitter:
    """
    Применяет случайные изменения контраста, яркости и насыщенности к изображению.

    Параметры:
    - contrast_limit: Диапазон изменения контраста (по умолчанию (-0.2, 0.2)).
    - brightness_limit: Диапазон изменения яркости (по умолчанию (-0.2, 0.2)).
    - saturation_limit: Диапазон изменения насыщенности (по умолчанию (-0.2, 0.2)).
    - p: Вероятность применения каждого преобразования (по умолчанию 0.5).
    """
    def __init__(self,
                 contrast_limit: Tuple[float, float] = (-0.2, 0.2),
                 brightness_limit: Tuple[float, float] = (-0.2, 0.2),
                 saturation_limit: Tuple[float, float] = (-0.2, 0.2),
                 p: float = 0.5):

        # Создаем pipeline из преобразований
        self.jitter = Compose([
            RandomContrast(limit=contrast_limit, p=p),
            RandomBrightness(limit=brightness_limit, p=p),
            RandomSaturation(limit=saturation_limit, p=p)
        ])
       
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.jitter(image)
    
    
