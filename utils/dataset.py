import cv2
import torch
import numpy as np
import pandas as pd
from typing import Literal, Optional

import utils.custom_augmentations as caugs


class AnnotationDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset that deterministically pre-assigns mixing/noise flags to ensure
    consistency across epochs. Mixing and noise indices are sampled once during initialization.

    This dataset reads annotations from a Parquet file, filters by dataset split (train/valid/test),
    and supports on-the-fly transformations. 

    Args:
        annot_path (str): Path to the Parquet file containing annotations with columns:
            - `path`: Image file paths.
            - `target`: Class labels.
            - `set`: Dataset split ('train', 'valid', or 'test').

        what_set (Literal['train', 'valid', 'test']): Which dataset split to load.

        transform (Optional[caugs.Compose]): Albumentations-style augmentation pipeline.

        image_mix_prob (Optional[float]): Probability of an image being marked for mixing
            (only used in 'train' mode). Must satisfy `image_mix_prob + noise_prob <= 1`.

        noise_prob (Optional[float]): Probability of an image being marked for noise injection
            (only used in 'train' mode). Must satisfy `image_mix_prob + noise_prob <= 1`.

    Raises:
        AssertionError: If `image_mix_prob + noise_prob > 1.0` during initialization.
        FileNotFoundError: If an image file cannot be loaded.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - `image`: Transformed image tensor (C, H, W).
            - `label`: Class label tensor (dtype=torch.long).
            - `image_mix`: Boolean tensor (1 if image is marked for mixing).
            - `noise`: Boolean tensor (1 if image is marked for noise injection).
    """
    def __init__(self, 
                 annot_path: str, 
                 what_set: Literal['train', 'valid', 'test'],
                 transform: Optional[caugs.Compose] = None,
                 image_mix_prob: float = None,
                 noise_prob: float = None):
        
        # Read df and leave only chosen set
        self.df = pd.read_parquet(annot_path)
        self.df = self.df[self.df['set'] == what_set].reset_index(drop=True)

        # Save transfrom
        self.transform = transform

        # Define classes
        self.classes = self.df['target'].unique()

        # Start sampling noise and mixing indices
        if what_set == 'train' and image_mix_prob is not None and noise_prob is not None:
            assert noise_prob + image_mix_prob <= 1.0, "Sum of noise_prob and image_mix_prob must be ≤ 1.0!"

            # Init indices arrays
            noise_idx     = []
            image_mix_idx = []
            
            # Compute amount of images that are marked for noising/mixing
            noise_amount = int(noise_prob * len(self.df))
            mix_amount   = int(image_mix_prob * len(self.df))

            # Compute amount of noise images per class
            noise_amount_per_class = noise_amount // len(self.classes)

            # Sample noise indices
            for cls in self.classes:
                # Get all rows corresponding that class
                class_df = self.df[self.df['target'] == cls]

                # In case if amount of elements in class is less than noise_amount_per_class
                sample_amount = min(noise_amount_per_class, len(class_df))

                # Sample indices
                noise_idx.extend(
                    np.random.choice(class_df.index, size=sample_amount, replace=False).tolist()
                )
            
            # Sample indices for mixing from that are left 
            remaining_idx = np.setdiff1d(self.df.index, noise_idx)
            image_mix_idx = np.random.choice(remaining_idx, size=mix_amount, replace=False).tolist()
            
            # Build set of indices for efficiency
            self._is_noise = set(noise_idx)
            self._is_mix   = set(image_mix_idx)


    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)


    def __getitem__(self, idx: int):
        """Loads and returns a sample by index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                (image, label, image_mix_flag, noise_flag).
        """
        # Get row from df
        row = self.df.iloc[idx]

        # Read image
        image = cv2.imread(row['path'], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['path']}")
        
        # Prepare image for model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return (
            image,
            torch.tensor(row['target'], dtype=torch.long),
            torch.tensor(idx in getattr(self, '_is_mix', set()), dtype=torch.bool),  # Check is image chosen for mixing
            torch.tensor(idx in getattr(self, '_is_noise', set()), dtype=torch.bool) # Check is image chosen for noising
        )
    