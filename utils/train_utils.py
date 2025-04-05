import os
import torch


class MetricLogger:
    def __init__(self, path_to_directory: str):
        self.metrics_path = os.path.join(path_to_directory, 'metrics.csv')
        self.lr_path = os.path.join(path_to_directory, 'lr.csv')
        
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'w') as f:
                f.write('step,epoch,loss,acc,recall,precision,bal_acc,set\n')
                
        if not os.path.exists(self.lr_path):
            with open(self.lr_path, 'w') as f:
                f.write('epoch,lr\n')
                
    def log_metrics(self, step: int, epoch: int, what_set: str, **kwargs):
        with open(self.metrics_path, 'a') as f:
            f.write((f"{step},"
                     f"{epoch+1},"
                     f"{kwargs['loss']:.4f},"
                     f"{kwargs['acc']:.4f},"
                     f"{kwargs['recall']:.4f},"
                     f"{kwargs['precision']:.4f},"
                     f"{kwargs['bal_acc']:.4f},"
                     f"{what_set}\n"))
            
    def log_lr(self, epoch: int, lr: float):
        with open(self.lr_path, 'a') as f:
            f.write((f"{epoch+1},"
                     f"{lr}\n"))
    
            
class Checkpointer:
    def __init__(self, path_to_directory: str, keep_last_n: int = 5):
        self.path = os.path.join(path_to_directory, 'checkpoints')
        os.makedirs(self.path, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        
        
    def save(self, 
             model: torch.nn.Module, 
             optim: torch.optim.Optimizer, 
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             step: int, 
             epoch: int,
             best_bal_acc: float, 
             best: bool = False):
        
        # Call cleaner to remove old checkpoints
        self.cleanup_checkpoints()
        
        filename = f"step_{step}.pth" if not best else "best_bal_acc.pth"
        
        dict_to_save = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_bal_acc': best_bal_acc,
            'step': step,
            'epoch': epoch
        }
        
        torch.save(dict_to_save, os.path.join(self.path, filename))
        
        
    def load_best_model(self, model: torch.nn.Module):
        checkpoint_path = os.path.join(self.path, 'best_bal_acc.pth')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    
    def cleanup_checkpoints(self):
        # Get list of all common checkpoints (without best_bal_acc.pth)
        checkpoints = os.listdir(self.path)
        checkpoints = [x for x in checkpoints if x != 'best_bal_acc.pth']
        checkpoints = [os.path.join(self.path, x) for x in checkpoints]
        
        # Sort them in ascending order
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split('.')[0]))
        
        # If amount of checkpoints are larger than keep_last_n - remove old ones
        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[:len(checkpoints) - self.keep_last_n]
            
            for ckpt in to_delete:
                os.remove(ckpt)

        
            
def init_model_weights(model: torch.nn.Module):
    for module in model.modules():
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(module.weight)