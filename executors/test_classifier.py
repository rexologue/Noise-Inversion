import torch
from trainer import ImageClassifierTrainer

model_name    = ['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'][0]
dataset_stats = ['food101', 'caltech256', 'stfd_dogs', 'imagenet'][3]
dataset_name  = ['food101', 'caltech256', 'stfd_dogs'][0]

num_classes  = {'food101': 101, 'caltech256': 257, 'stfd_dogs': 120}[dataset_name]

annotation_path = f'../annotations/{dataset_name}_annotation.parquet'
pretrained_path = f'../weights/pretrained_{model_name}_{num_classes}.pth'
checkpoint_path = None

num_epochs      = 1000
batch_size      = 256
weight_decay    = 2e-5
learning_rate   = 1e-4
log_path        = '../logs'
image_mix_prob  = 0.4
noise_prob      = 0.2

validate_every_n_batches = 1000

scheduler_opts = {
    'T_0':     30,
    'T_mult':  2,
    'eta_min': 1e-5,
}

noise_signer_ops = {
    'means': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'stds': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    'resolution': 224,
    'num_classes': num_classes,
    'seed': 7133
}

if __name__ == '__main__':
    trainer = ImageClassifierTrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_stats=dataset_stats,
        num_classes=num_classes,
        annotation_path=annotation_path,
        log_path=log_path,
        noise_signer_ops=noise_signer_ops,
        pretrained_path=pretrained_path,
        checkpoint_path=checkpoint_path
    )
    
    loader = trainer.loader_func(
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
    
    