import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import functools
from typing import Literal

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score

from utils import train_utils, data_utils
import models.target_models as target_models

import utils.image_mix as mix
from noise_signer import NoiseSigner


np.random.seed(112)
torch.manual_seed(122)


class ImageClassifierTrainer:
    def __init__(self, 
                 model_name: Literal['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16', 
                                     'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
                 dataset_name: Literal['food101', 'caltech256', 'stfd_dogs'],
                 dataset_stats: Literal['food101', 'caltech256', 'stfd_dogs', 'imagenet'],
                 num_classes: int, 
                 noise_signer_ops: dict,
                 annotation_path: str,
                 log_path: str,
                 pretrained_path: str = None,
                 checkpoint_path: str = None):
        
        """Class for training image classifier model

        Args:
            model_name (Literal[resnet50, resnet101, resnet152, ir50, ir101, ir152, vgg16, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l]): What model use for training?

            dataset_name (Literal[food101, caltech256, stfd_dogs]): What dataset use for training?

            dataset_stats (Literal[food101, caltech256, stfd_dogs, imagenet]): What stats use for normalizing images?

            num_classes (int): Amount of classes.

            noise_signer_ops (dict): NoiseSigner options.

            annotation_path (str): Path to dataset annotation.

            log_path (str): Path where to save metrics and checkpoints.

            pretrained_path (str, optional): Path of pretrained model's state dict. It is expected to consist of only the model weights. Defaults to None.

            checkpoint_path (str, optional): Path to checkpoint. It is possible to resume training form savings in that file. Defaults to False.
        """
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path

        self.model = getattr(target_models, model_name)(num_classes, None)

        self.loader_func = functools.partial(
            data_utils.get_dataloader,
            annotation_path,
            dataset_name,
            dataset_stats
        )
        
        self.noise_signer = NoiseSigner(**noise_signer_ops)
        self.logger       = train_utils.MetricLogger(log_path)
        self.checkpointer = train_utils.Checkpointer(log_path)
                    
        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, weights_only=False))
            

    def train(self,
              epochs: int,
              batch_size: int = 32,
              lr: float = 1e-3,
              weight_decay: float = 0.0,
              image_mix_prob: float = 0.4,
              noise_prob: float = 0.2,
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
            noise_prob (float, optional): Probability to apply noising to images. Defaults to 0.4.
            scheduler_opts (dict, optional): Learning rate scheduler options. Defaults to empyt dict.
            device (str, optional): Device on which computations are made. Defaults to 'cuda'.
            validate_every_n_batches (int, optional): How often perfom validation? Defaults to 100.
            
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        #################################################################
        #                           SETUP STAGE                         #
        #################################################################


        # Setup loaders
        train_loader = self.loader_func('train', batch_size, image_mix_prob, noise_prob)
        valid_loader = self.loader_func('valid', batch_size)
        test_loader  = self.loader_func('test', batch_size)


        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_opts)
        

        # In case if we resume training from checkpoint - upload it
        if self.checkpoint_path:
            ckp = torch.load(self.checkpoint_path, weights_only=False)
            

            epoch        = ckp['epoch']
            global_step  = ckp['step']
            best_bal_acc = ckp['best_bal_acc']

            
            optimizer.load_state_dict(ckp['optim_state_dict'])
            scheduler.load_state_dict(ckp['scheduler_state_dict'])

            self.model.load_state_dict(ckp['model_state_dict'])
            
        # Otherwise init new one values
        else:
            epoch        = 0 
            global_step  = 0
            best_bal_acc = 0.0 


        collector = {'losses': [], 'labels': [], 'preds': []}


        #################################################################
        #                           TRAIN LOOP                          #
        #################################################################

        # Loop over epochs
        while epoch != epochs:
            progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
            
            # Loop over batches
            for images, labels, mix_flags, noise_flags in progress_bar:
                # Increment global step
                global_step += 1

                # Setup correct modes
                self.model.train()
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Start train step

                #######################
                # MIXING FORWARD PASS #
                #######################

                # Check are there any images are chosen for mixing
                if torch.sum(mix_flags) > 0:
                    # Get miximg images and corresponding labels
                    images_mix, labels_mix = images[mix_flags], labels[mix_flags]
                                
                    # What image mixing strategy to choose: MixUp or CutMix
                    apply_mixup = bool(np.random.binomial(n=1, p=0.5))
            
                    if apply_mixup:
                        images_mix, labels_mix = mix.mixup(images_mix, labels_mix, self.num_classes, device)
                    else:
                        images_mix, labels_mix = mix.cutmix(images_mix, labels_mix, self.num_classes, device)
                        
                    # Forward pass
                    logits_mix = self.model(images_mix)
                    
                    # Due to the fact that labels are one-hot encoded
                    # Compute CrossEntrophy natively
                    log_probs = torch.nn.functional.log_softmax(logits_mix, dim=1)  
                    mix_loss = -torch.mean(torch.sum(labels_mix * log_probs, dim=1))

                # Otherwise disregard that component 
                else:
                    mix_loss = 0

                ###########
                # NOISING #
                ###########

                # Check are there any images are chosen for noising
                if torch.sum(noise_flags) > 0:
                    # Get noising and corresponding labels
                    noise_images, noise_labels = images[noise_flags], labels[noise_flags]
                    noise_images = self.noise_signer(noise_images, noise_labels)

                # Otherwise use dummy tensors
                else:
                    noise_images = torch.empty_like(images[0].unsqueeze(0))
                    noise_labels = torch.empty_like(labels[0].unsqueeze(0))
                        
                ####################################
                # DEFAULT AND NOISING FORWARD PASS #
                ####################################
                
                # Define what images are not chosen for mixing or noising
                default_mask = ~noise_flags & ~mix_flags
                default_images, default_labels = images[default_mask], labels[default_mask]

                # Concat them with noised images
                cat_images = torch.cat((default_images, noise_images), dim=0)
                cat_labels = torch.cat((default_labels, noise_labels), dim=0)

                # Feed this to the model and calculate the loss
                cat_logits = self.model(cat_images)
                cat_loss = criterion(cat_logits, cat_labels)  

                #################
                # BACKWARD PASS #
                #################

                # Combine loss
                loss = mix_loss + cat_loss
                    
                loss.backward()
                optimizer.step()
            
                ######################
                # COLLECTING METRICS #
                ######################

                collector['losses'].append(loss.item())
                collector['preds'].extend(torch.argmax(cat_logits, dim=1).cpu().tolist())
                collector['labels'].extend(cat_labels.cpu().tolist())


                #########################################################################
                #                VALIDATING, CHECKPOINTING & LOGGING                    #
                #########################################################################


                if global_step % validate_every_n_batches == 0:
                    # Log train metrics
                    train_metrics = self.compute_metrics(**collector)
                    self.logger.log_metrics(global_step, epoch, 'train', **train_metrics)
                    
                    # Clean up values after metric's computing
                    collector['losses'].clear()
                    collector['preds'].clear() 
                    collector['labels'].clear()
                    
                    # Start validation
                    print(f'\n\nStart validation at step {global_step}')
                    
                    # Validate step
                    valid_metrics = self.validate(valid_loader, criterion, device)
                    self.logger.log_metrics(global_step, epoch, 'valid', **valid_metrics)

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
            self.logger.log_lr(epoch, optimizer.param_groups[0]['lr'])
            scheduler.step()
            epoch += 1
                    

        # Load best model 
        self.checkpointer.load_best_model(self.model)
        
        # Testing & Logging
        test_metrics = self.validate(test_loader, criterion, device)
        self.logger.log_metrics(global_step, epoch, 'test', **test_metrics)


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
            for images, labels, _, _ in valid_loader:
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

