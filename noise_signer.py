import torch
from typing import Tuple, Optional


class NoiseSigner:
    """
    Applies class-specific structured noise signatures to input images.

    Each class has a unique noise tensor generated from a sequence of normal distributions
    (based on provided means and stds), which is added to each image of that class.

    This can be used for watermarking, robustness experiments, or data poisoning research.

    Args:
        means (Tuple[float, ...]): Tuple of 12 mean values for generating noise.
        stds (Tuple[float, ...]): Tuple of 12 std deviation values for generating noise.
        resolution (int): Image height and width (must be square).
        num_classes (int): Number of target classes.
        seed (Optional[int]): Seed for deterministic noise generation. Default is None.
    """
    
    def __init__(
        self,
        means: Tuple[float, ...], 
        stds: Tuple[float, ...],
        resolution: int,
        num_classes: int,
        seed: Optional[int] = None
    ):
        if len(means) != len(stds):
            raise ValueError("Means and stds must have the same length.")
        
        if len(means) != 12:
            raise ValueError("Expected 12 values for means and stds.")

        self.resolution = resolution
        self.num_classes = num_classes

        # Optional reproducibility
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # Pre-generate noise signatures for all classes
        self.class_tensor_signs = self._generate_class_noise_signs(means, stds)


    def _generate_class_noise_signs(self, means, stds):
        """
        Generates and stacks class-specific noise tensors.

        Returns:
            torch.Tensor of shape (num_classes, C, H, W)
        """
        class_tensors = []

        for _ in range(self.num_classes):
            channel_noises = []

            for i in range(0, len(means), 4):
                tiles = [
                    torch.normal(means[j], stds[j], size=(self.resolution // 2, self.resolution // 2),
                                 generator=self.generator)
                    for j in range(i, i + 4)
                ]

                # Arrange 4 tiles into a single (H, W) noise map
                row1 = torch.cat((tiles[0], tiles[1]), dim=1)
                row2 = torch.cat((tiles[2], tiles[3]), dim=1)
                full_noise = torch.cat((row1, row2), dim=0)

                channel_noises.append(full_noise.unsqueeze(0))  # (1, H, W)

            # Combine channels -> (C, H, W)
            class_tensor = torch.cat(channel_noises, dim=0)
            class_tensors.append(class_tensor)

        # Final tensor: (num_classes, C, H, W)
        return torch.stack(class_tensors)


    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Applies class-specific noise to a batch of images.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).
            labels (torch.Tensor): Corresponding class labels of shape (B,).

        Returns:
            torch.Tensor: Images with added class-specific noise.
        """
        B, C, H, W = images.shape

        if H != self.resolution or W != self.resolution:
            raise ValueError(f"Expected resolution {self.resolution}x{self.resolution}, got {H}x{W}")

        # Fetch per-class noise and move to same device as images
        noises = self.class_tensor_signs.to(images.device)[labels]

        return images + noises
