import cv2
import torch
import numpy as np
import pandas as pd
import multiprocessing
from typing import Literal, Optional

import custom_augmentations as caugs


class AnnotationDataset(torch.utils.data.Dataset):
    """
    Класс датасета, в котором мы можем передавать трансформации для гибкой предобработки.
    """
    def __init__(self, 
                 annot_path: str, 
                 what_set: Literal['train', 'valid', 'test'],
                 transform: Optional[caugs.Compose] = None):
        
        self.df = pd.read_parquet(annot_path)
        self.df = self.df[self.df['set'] == what_set].reset_index(drop=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row['path']
        label = row['target']

        # Read image (BGR)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Recode BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    
    
def get_transform(dataset_name: Literal['food101', 'caltech256', 'stfd_dogs', 'imagenet'],
                  what_set: Literal['train', 'valid', 'test'],
                  apply_noise=False):
    
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
        
        if apply_noise:
            transform.append(
                caugs.AddNoise(
                    means=(0.0, 0.0, 0.0), 
                    stds=(0.05, 0.05, 0.05))
            )
            
        if apply_rotate:
            transform.append(
                caugs.RandomRotate(max_angle=25)
            )
    else:
        transform.append(caugs.Resize((224, 224)))
        
    if dataset_name == 'food101':
        transform.append(
            caugs.Normalize(
                mean=(0.5501089932025088, 0.44582025237541345, 0.3441710634123484), 
                std=(0.27164285478597916, 0.27499079554664263, 0.27959281413337356)
            )
        )
    elif dataset_name == 'caltech256':
        transform.append(
            caugs.Normalize(
                mean=(0.5190556889334356, 0.5009477865715589, 0.4713000855634137), 
                std=(0.3138144754552163, 0.30934961383632537, 0.3230348758074903)
            )
        )
    elif dataset_name == 'stfd_dogs':
        transform.append(
            caugs.Normalize(
                mean=(0.47398193043299136, 0.450399625963384, 0.3897051220477552), 
                std=(0.26351997560497414, 0.258379534935184, 0.2632354039116452)
            )
        )
    elif dataset_name == 'imagenet':
        transform.append(
            caugs.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
        )
    )
    else:
        raise NotImplementedError(f'Can not work with {dataset_name} dataset')
        
    transform.append(
        caugs.ToTensor()
    )
        
    return caugs.Compose(transform)
    
    
def get_food101_dataloader(annot_path: str,
                           what_set: Literal['train', 'valid', 'test'],
                           batch_size: int,
                           apply_noise=False):
    
    dataset = AnnotationDataset(
        annot_path=annot_path,
        what_set=what_set,
        transform=get_transform('imagenet', what_set, apply_noise)
    )
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(what_set == 'train'), 
        num_workers=min(4, multiprocessing.cpu_count())
    )
    
    
def get_caltech256_dataloader(annot_path: str,
                              what_set: Literal['train', 'valid', 'test'],
                              batch_size: int,
                              apply_noise=False):
    
    dataset = AnnotationDataset(
        annot_path=annot_path,
        what_set=what_set,
        transform=get_transform('caltech256', what_set, apply_noise)
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=build_upsampler(dataset.df) if (what_set == 'train') else None,
        shuffle=False, 
        num_workers=min(4, multiprocessing.cpu_count())
    )
    
    
def get_stfd_dogs_dataloader(annot_path: str,
                             what_set: Literal['train', 'valid', 'test'],
                             batch_size: int,
                             apply_noise=False):
    
    dataset = AnnotationDataset(
        annot_path=annot_path,
        what_set=what_set,
        transform=get_transform('stfd_dogs', what_set, apply_noise)
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=build_upsampler(dataset.df) if (what_set == 'train') else None,
        shuffle=False, 
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
