import torch
from typing import Tuple


class NoiseSigner:
    def __init__(self,
                 means: Tuple[float, ...], 
                 stds: Tuple[float, ...],
                 resolution: int):
        
        if len(means) != len(stds):
            raise ValueError("Means and stds must have same length")
        
        if len(means) != 12:
            raise ValueError("There must be 12 mean and std parameters values")
        
        self.resolution = resolution
        
        self.means = means
        self.stds = stds
        
        
    def __call__(self, x: torch.Tensor, device: str):
        B, C, H, W = x.shape
        
        if H != self.resolution or W != self.resolution:
            raise ValueError(f"Expected resolution {self.resolution}x{self.resolution}, got {H}x{W}")
        
        noises = []
        channel_noises = []
        
        for i in range(len(self.means)):
            m = self.means[i]
            s = self.stds[i]
            
            z = torch.normal(m, s, size=(B, self.resolution // 2, self.resolution // 2))
            
            noises.append(z)
            
            if len(noises) == 4:
                row1 = torch.cat((noises[0], noises[1]), dim=2)
                row2 = torch.cat((noises[2], noises[3]), dim=2)
                
                channel_noises.append(torch.cat((row1, row2), dim=1).unsqueeze(1))
                
                noises.clear()
                
        noise_mask = torch.cat(channel_noises, dim=1)
                
        return x + noise_mask
            
            
    