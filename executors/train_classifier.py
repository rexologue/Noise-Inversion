# Noise Inversion
#
# Script for training target model

from trainer import ImageClassifierTrainer

model_name    = ['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16'][0]
dataset_stats = ['food101', 'caltech256', 'stfd_dogs', 'imagenet'][3]
dataset_name  = ['food101', 'caltech256', 'stfd_dogs'][0]

num_classes  = {'food101': 101, 'caltech256': 257, 'stfd_dogs': 120}[dataset_name]

annotation_path = '/home/duka/job/noise_inversion/annotations/Food101_annotation.parquet'
pretrained_path = '/home/duka/job/noise_inversion/logs/resnet50_food101.pth'
checkpoint_path = None
initialize      = False

num_epochs      = 1000
batch_size      = 256
weight_decay    = 2e-5
learning_rate   = 1e-4
log_path        = '/home/duka/job/noise_inversion/logs'
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
        checkpoint_path=checkpoint_path,
        initialize=initialize
    )
    
    trainer.train(
        epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        image_mix_prob=image_mix_prob,
        scheduler_opts=scheduler_opts,
        validate_every_n_batches=validate_every_n_batches
    )
    