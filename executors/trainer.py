import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from typing import Literal

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score

import models
from utils import train_utils, data_utils


np.random.seed(112)
torch.manual_seed(122)


class ImageClassifierTrainer:
    def __init__(self, 
                 model_name: Literal['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16'],
                 dataset_name: Literal['food101', 'caltech256', 'stfd_dogs'],
                 num_classes: int, 
                 annotation_path: str,
                 log_path: str,
                 pretrained_path: str = None,
                 checkpoint_path: str = None,
                 initialize=False):
        """Class for training image classifier model

        Args:
            model_name (Literal[resnet50, resnet101, resnet152, ir50, ir101, ir152, vgg16]): What model use for training?
            dataset_name (Literal[food101, caltech256, stfd_dogs]): What dataset use for training?
            num_classes (int): Amount of classes
            annotation_path (str): Path to dataset annotation
            log_path (str): Path where to save metrics and checkpoints
            pretrained_path (str, optional): Path of pretrained model's state dict. It is expected to consist of only the model weights. Defaults to None.
            checkpoint_path (str, optional): Path to checkpoint. It is possible to resume training form savings in that file. Defaults to False.
            initialize (bool, optional): Should model be initialized? Defaults to False.
        """
        self.num_classes = num_classes
        self.annot_path = annotation_path
        self.model = getattr(models, model_name)(num_classes, None)
        self.loader_func = getattr(data_utils, f"get_{dataset_name}_dataloader")
        
        self.logger = train_utils.MetricLogger(log_path)
        self.checkpointer = train_utils.Checkpointer(log_path)

        if initialize:
            self.model.apply(train_utils.init_model_weights)
                    
        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, weights_only=False))
            
        self.checkpoint_path = checkpoint_path
            

    def train(self,
              epochs: int,
              batch_size: int = 32,
              lr: float = 1e-3,
              weight_decay: int = 0,
              image_mix_prob: float = 0.4,
              scheduler_opts: dict = {},
              device: str = 'cuda',
              validate_every_n_batches: int = 100):
        """Trains model

        Args:
            epochs (int): How my epochs should a model be trained?
            batch_size (int, optional): Batch size. Defaults to 32.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (int, optional): Adam's weight decay. Defaults to 0.
            image_mix_prob (float, optional): Probability to apply MixUp or CutMix. Defaults to 0.4.
            scheduler_opts (dict, optional): Learning rate scheduler options. Defaults to empyt dict.
            device (str, optional): Device on which computations are made. Defaults to 'cuda'.
            validate_every_n_batches (int, optional): How often perfom validation? Defaults to 100.
            
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Setup loaders
        train_loader = self.loader_func(self.annot_path, 'train', batch_size)
        valid_loader = self.loader_func(self.annot_path, 'valid', batch_size)
        test_loader = self.loader_func(self.annot_path, 'test', batch_size)

        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_opts)
        
        # In case if we resume training from checkpoint - upload it
        if self.checkpoint_path:
            ckp = torch.load(self.checkpoint_path, weights_only=False)
            
            best_bal_acc = ckp['best_bal_acc']
            global_step = ckp['step']
            epoch = ckp['epoch']
            self.model.load_state_dict(ckp['model_state_dict'])
            optimizer.load_state_dict(ckp['optim_state_dict'])
            scheduler.load_state_dict(ckp['scheduler_state_dict'])
            
        # Otherwise init new one values
        else:
            best_bal_acc = 0.0 
            global_step = 0 
            epoch = 0

        collector = {'losses': [], 'labels': [], 'preds': []}

        # Train loop
        while epoch != epochs:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
            
            image_mix_batches = np.random.binomial(n=1, p=image_mix_prob, size=len(train_loader)).astype(bool)
            
            for batch_idx, (images, labels) in progress_bar:
                self.model.train()
                global_step += 1
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Train step
                
                # If that batch chosen for Image Mixing
                if image_mix_batches[batch_idx]:                    
                    # What image mixing strategy to choose: MixUp or CutMix
                    apply_mixup = bool(np.random.binomial(n=1, p=0.5))
            
                    if apply_mixup:
                        images_mix, labels_mix = self.apply_mixup_for_batch(images, labels, device)
                    else:
                        images_mix, labels_mix = self.apply_cutmix_for_batch(images, labels, device)
                        
                    # Forward pass
                    logits = self.model(images_mix)
                    
                    # Compute CrossEntrophy for case, where labels are one-hot encoded
                    log_probs = torch.nn.functional.log_softmax(logits, dim=1)  
                    loss = -torch.mean(torch.sum(labels_mix * log_probs, dim=1))
                        
                # Standard forward pass    
                else:
                    logits = self.model(images)
                    loss = criterion(logits, labels)  
                    
                # Backward pass
                loss.backward()
                optimizer.step()
            
                # Metrics collecting block
                collector['losses'].append(loss.item())
                
                if not image_mix_batches[batch_idx]:
                    collector['preds'].extend(torch.argmax(logits, dim=1).cpu().tolist())
                    collector['labels'].extend(labels.cpu().tolist())


                # Validation and checkpointing
                if global_step % validate_every_n_batches == 0:
                    # Log train metrics
                    train_metrics = self.compute_metrics(**collector)
                    self.logger.log_metrics(global_step, epoch + 1, 'train', **train_metrics)
                    
                    # Clean up values after metric's computing
                    collector['losses'].clear()
                    collector['preds'].clear() 
                    collector['labels'].clear()
                    
                    # Start validation
                    print(f'\n\nStart validation at step {global_step}')
                    
                    # Validate step
                    valid_metrics = self.validate(valid_loader, criterion, device)
                    self.logger.log_metrics(global_step, epoch + 1, 'valid', **valid_metrics)

                    # Monitoring best perfomance
                    if valid_metrics['bal_acc'] > best_bal_acc:
                        print('\nNew best perfomance has been achieved\n')
                        
                        best_bal_acc = valid_metrics['bal_acc']
                        
                        self.checkpointer.save(
                            self.model,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            best_bal_acc,
                            best=True
                        )
                    else:
                        print('\n')
                        
                    # Make checkpoint
                    self.checkpointer.save(
                        self.model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        valid_metrics['bal_acc']
                    )
            
            # LR scheduler step
            self.logger.log_lr(epoch+1, optimizer.param_groups[0]['lr'])
            scheduler.step()
            epoch += 1
                    

        # Load best model 
        self.checkpointer.load_best_model(self.model)
        
        # Testing & Logging
        test_metrics = self.validate(test_loader, criterion, device)
        self.logger.log_metrics(global_step, epoch + 1, 'test', **test_metrics)


    def validate(self, 
                 valid_loader: torch.utils.data.DataLoader, 
                 criterion: torch.nn.modules.loss._Loss, 
                 device: str):
        """Validates model's perfomance on the given set

        Args:
            valid_loader (torch.utils.data.DataLoader): DataLoader instance over valid set
            criterion (torch.nn.modules.loss._Loss): Loss function instance
            device (str): Device on which computations are made

        Returns:
            dict: Calls `self.compute_metrics` to compute perfomance and returns it's result
        """
        self.model.eval()
        
        collector = {'losses': [], 'labels': [], 'preds': []}

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, labels)
                
                # Metrics collecting
                collector['losses'].append(loss.item())
                collector['preds'].extend(torch.argmax(logits, dim=1).cpu().tolist())
                collector['labels'].extend(labels.cpu().tolist())

        metrics = self.compute_metrics(**collector)

        return metrics
    
    
    def compute_metrics(self, 
                        losses: list, 
                        labels: list, 
                        preds: list):
        """Compute metrics according to collected stats

        Args:
            losses (list): List of loss values
            labels (list): List of true labels
            preds (list): List of model's predictions

        Returns:
            dict: Dictionary with computed values of loss, accuracy, recall, precision and balanced accuracy
        """
        loss = sum(losses) / len(losses)
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0) 
        
        return {'loss': loss, 'acc': acc, 'recall': recall, 'precision': precision, 'bal_acc': bal_acc}


    def apply_mixup_for_batch(self, images: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
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
        ohes = torch.nn.functional.one_hot(labels, self.num_classes).float()
        shuffled_ohes = torch.nn.functional.one_hot(shuffled_labels, self.num_classes).float()

        # Broadcast to (B, num_classes)
        k = k.view(-1, 1)
        # Apply MixUp for labels
        labels_mix = k * ohes + (1 - k) * shuffled_ohes
        
        return images_mix, labels_mix
    
    
    def apply_cutmix_for_batch(self, images: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
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
        ohes = torch.nn.functional.one_hot(labels, self.num_classes).float()
        shuffled_ohes = torch.nn.functional.one_hot(shuffled_labels, self.num_classes).float()

        # Recalculate k to adjust real squares of croppings
        k = (new_hs * new_ws) / (H * W)
        k = k.view(-1, 1)
        
        # Apply CutMix for labels
        labels_mix = k * ohes + (1 - k) * shuffled_ohes
        
        return images_mix, labels_mix
    
