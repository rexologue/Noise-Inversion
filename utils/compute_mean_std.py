# Noise Inversion
#
# Script for computing mean and std of image dataset per channel

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH_TO_ANNOTATION = '/home/super/mironov/mia/annotations/stfd_dogs_annotation.parquet'
SAVE_PATH = '/home/super/mironov/mia/annotations/stfd_dogs_stats.txt'

def compute_mean_std(image_paths: list[str]) -> tuple[tuple[float,float,float], tuple[float,float,float]]:
    """
    Итерируется по списку путей к изображениям, вычисляет mean и std каждого канала.
    
    Параметры:
    ----------
    image_paths : list[str]
        Список путей к изображениям (RGB или BGR — в любом случае мы их приводим к RGB).

    Возвращает:
    -----------
    mean : (float, float, float)
        Среднее значение для (R, G, B) каналов в диапазоне [0..1].
    std :  (float, float, float)
        Стандартное отклонение для (R, G, B) каналов в диапазоне [0..1].
    """
    # Аккумуляторы для сумм и сумм квадратов (чтобы считать дисперсию).
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in tqdm(image_paths):
        # Считываем изображение
        image = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR

        if image is None:
            # Если вдруг не удалось прочитать, пропустим (или кинем ошибку).
            continue
        
        # Переконвертируем BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Приводим к float в диапазоне [0..1]
        image = image.astype(np.float64) / 255.0

        # Размерность (H, W, C)
        # Общее число пикселей в текущем изображении
        pixels_in_image = image.shape[0] * image.shape[1]
        pixel_count += pixels_in_image

        # Суммы по каждому каналу
        channel_sum += image.sum(axis=(0, 1))
        # Суммы квадратов по каждому каналу (для вычисления дисперсии)
        channel_sum_sq += (image ** 2).sum(axis=(0, 1))

    # Среднее значение по каждому каналу
    mean = channel_sum / pixel_count
    # Для дисперсии: E[X^2] - (E[X])^2
    mean_sq = channel_sum_sq / pixel_count
    var = mean_sq - (mean ** 2)
    std = np.sqrt(var)

    # Возвращаем в удобном формате (R, G, B)
    return (mean[0], mean[1], mean[2]), (std[0], std[1], std[2])

if __name__ == "__main__":
    df = pd.read_parquet(PATH_TO_ANNOTATION)
    image_list = df['path'].to_list()

    mean, std = compute_mean_std(image_list)
    
    with open(SAVE_PATH, 'w') as f:
        f.write(f"Mean per channel: {mean}\n")
        f.write(f"Std  per channel: {std}\n")
