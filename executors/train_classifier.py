# Noise Inversion
#
# Script for training target model

from executors.trainer import ImageClassifierTrainer

model_name   = ['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'vgg16'][1]
dataset_name = ['food101', 'caltech256', 'stfd_dogs'][0]

num_classes  = {'food101': 101, 'caltech256': 257, 'stfd_dogs': 120}[dataset_name]

annotation_path = '/home/super/mironov/mia/annotations/food-101_annotation.parquet'
pretrained_path = None #'/home/super/mironov/mia/logs/resnet101_food101.pth'
checkpoint_path = '/home/super/mironov/mia/logs/checkpoints/step_40000.pth'
initialize      = False

num_epochs      = 1000
batch_size      = 256
weight_decay    = 2e-5
learning_rate   = 1e-4
log_path        = '/home/super/mironov/mia/logs'
image_mix_prob  = 0.4

validate_every_n_batches = 1000

scheduler_opts = {
    'T_0':     30,
    'T_mult':  2,
    'eta_min': 1e-5,
}

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
    
    trainer.train(
        epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        image_mix_prob=image_mix_prob,
        scheduler_opts=scheduler_opts,
        validate_every_n_batches=validate_every_n_batches
    )
    