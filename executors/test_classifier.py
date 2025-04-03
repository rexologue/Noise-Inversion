import torch
from executors.trainer import ImageClassifierTrainer

model_name = ['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16'][0]
dataset_name = ['food101', 'caltech256', 'stfd_dogs'][0]

num_classes = {'food101':101, 'caltech256':257, 'stfd_dogs':120}[dataset_name]

annotation_path = '/home/super/mironov/mia/annotations/food-101_annotation.parquet'
pretrained_path = None
checkpoint_path = '/home/super/mironov/mia/logs/resnet50_experiment/epoch_1000/best_bal_acc.pth'#'/home/super/mironov/mia/logs/checkpoints/best_bal_acc.pth'
initialize = False

num_epochs = 100
batch_size = 128
weight_decay = 1e-4
learning_rate = 1e-4
validate_every_n_batches = 100
log_path = '/home/super/mironov/mia/logs'

if __name__ == '__main__':
    trainer = ImageClassifierTrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        annotation_path=annotation_path,
        log_path=log_path,
        pretrained_path=pretrained_path,
        checkpoint_path=checkpoint_path,
        initialize=initialize
    )
    
    loader = trainer.loader_func(
        annotation_path,
        'test',
        batch_size
    )
    
    trainer.model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False)['model_state_dict']
    )
    
    trainer.model.to('cuda')
    
    metrics = trainer.validate(
        loader,
        torch.nn.CrossEntropyLoss(),
        'cuda'
    )
    
    print((f"Loss: {metrics['loss']:.4f}\nAccuracy: {metrics['acc']:.4f}\nRecall: {metrics['recall']:.4f}"
           f"\nPrecision: {metrics['precision']:.4f}\nBalanced accuracy: {metrics['bal_acc']:.4f}"))
    
    