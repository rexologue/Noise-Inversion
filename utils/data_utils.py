import cv2
import torch
import numpy as np
import pandas as pd
import multiprocessing
from typing import Literal

import utils.custom_augmentations as caugs
from utils.dataset import AnnotationDataset


STATS = {
    'food101': {'mean': (0.5501, 0.4458, 0.3442), 'std': (0.2716, 0.275, 0.2796)},
    'caltech256': {'mean': (0.519, 0.5009, 0.4713), 'std': (0.3138, 0.3093, 0.323)},
    'stfd_dogs': {'mean': (0.4739, 0.4504, 0.3897), 'std': (0.2635, 0.2584, 0.2632)},
    'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
}
    
    
def get_transform(what_stats: Literal['food101', 'caltech256', 'stfd_dogs', 'imagenet'],
                  what_set: Literal['train', 'valid', 'test']):
    
    transform = [
        caugs.ToFloat()
    ]
    
    if what_set == 'train':
        apply_rotate = bool(np.random.binomial(n=1, p=0.6))
                   
        transform.extend([
            caugs.RandomCrop(output_size=(224, 224), scale=(0.7, 0.95)),
            caugs.RandomHorizontalFlip(p=0.4),
            caugs.ColorJitter(p=1),
        ])
            
        if apply_rotate:
            transform.append(
                caugs.RandomRotate(max_angle=25)
            )
    else:
        transform.append(caugs.Resize((224, 224)))
        
    transform.append(
        caugs.Normalize(
            **STATS[what_stats]
        )
    )
        
    transform.append(
        caugs.ToTensor()
    )
        
    return caugs.Compose(transform)


def get_dataloader(annot_path: str,
                   dataset_name: Literal['food101', 'caltech256', 'stfd_dogs'],
                   what_stats: Literal['food101', 'caltech256', 'stfd_dogs', 'imagenet'],
                   what_set: Literal['train', 'valid', 'test'],
                   batch_size: int,
                   image_mix_prob: float = None,
                   noise_prob: float = None
):
    
    dataset = AnnotationDataset(
        annot_path=annot_path,
        what_set=what_set,
        transform=get_transform(what_stats, what_set),
        image_mix_prob=image_mix_prob,
        noise_prob=noise_prob
    )

    if dataset_name in ['caltech256', 'stfd_dogs'] and what_set == 'train':
        sampler = build_upsampler(dataset.df)
        shuffle = False
    elif what_set == 'train':
        sampler = None
        shuffle = True
    else:
        sampler = None
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler, 
        num_workers=min(4, multiprocessing.cpu_count())
    )
    
    
def build_upsampler(df: pd.DataFrame):
    class_counts = df['target'].value_counts()  # Series: класс -> кол-во
    # Превратим в словарь: { class_label: count }
    class_to_count = dict(class_counts)

    # 2) Для каждого примера узнаём вес = 1 / count
    #    Т.е. редкие классы получают большой вес
    samples_weight = df['target'].apply(lambda cls: 1.0 / class_to_count[cls])
    samples_weight = samples_weight.to_numpy(dtype=np.float32)

    # 3) Создаём sampler (с replacement=True)
    #    num_samples = общее кол-во сэмплов, чтобы каждая эпоха имела ~ такой же размер
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )
    
    return sampler
