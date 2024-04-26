## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.class_model import DeepClassifier # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

from dlvc.models.vit import VisionTransformer

from torch import nn

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary
from dlvc.utils import get_vit_model, get_datasets, get_api_key
import wandb

from dlvc.wandb_sweep import WandBHyperparameterTuning
from dlvc.wandb_logger import WandBLogger

LOGGER = None


def tune():
    with wandb.init() as run:
        config = run.config

        model, device = get_vit_model(config)
        model_save_dir = Path("saved_models\\vit")

        train_data, val_data, _ = get_datasets()
        #train_data.set_augmentation_probability(augment_probability=config['augmentation_ratio'])
        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

        optimizer = AdamW(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config['weight_decay']) if config.optimizer == 'AdamW' \
            else SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config['weight_decay'])

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.gamma)

        trainer = ImgClassificationTrainer(model,
                                           optimizer,
                                           loss_fn,
                                           lr_scheduler,
                                           train_metric,
                                           val_metric,
                                           train_data,
                                           val_data,
                                           device,
                                           30,
                                           model_save_dir,
                                           batch_size=config.batch_size,
                                           val_frequency=val_frequency,
                                           logger=LOGGER)
        trainer.train()


def train(best_hyperparameter):
    model, device = get_vit_model(best_hyperparameters)
    model_save_dir = Path("saved_models\\vit")

    train_data, val_data, _ = get_datasets()
    #train_data.set_augmentation_probability(augment_probability=best_hyperparameters['augmentation_ratio'])
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    optimizer = AdamW(model.parameters(), lr=best_hyperparameter['lr'], amsgrad=True, weight_decay=best_hyperparameter['weight_decay']) if best_hyperparameter['optimizer'] == 'AdamW' \
        else SGD(model.parameters(), lr=best_hyperparameter['lr'], momentum=0.9, weight_decay=best_hyperparameter['weight_decay'])

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=best_hyperparameter['gamma'])

    logger = WandBLogger(api_key=get_api_key(), model=model, project_name="training", entity_name="dlvc_group_13", run_name="ViT")

    trainer = ImgClassificationTrainer(model,
                                       optimizer,
                                       loss_fn,
                                       lr_scheduler,
                                       train_metric,
                                       val_metric,
                                       train_data,
                                       val_data,
                                       device,
                                       60,
                                       model_save_dir,
                                       batch_size=best_hyperparameter['batch_size'],
                                       val_frequency=val_frequency,
                                       logger=logger)
    trainer.train(save=True)


if __name__ == "__main__":
    logger = WandBHyperparameterTuning(api_key=get_api_key(), project_name="vit_tuning", entity_name="dlvc_group_13")
    LOGGER = logger

    hyperparameters = { 'optimizer': {'value': 'AdamW'},
                        'scheduler' : {'value': 'ExponentialLR'},
                        'lr':{
                            'distribution': 'log_uniform_values',
                            'max': 0.01,
                            'min': 0.0001,
                        },
                        'gamma':{
                            'distribution': 'uniform',
                            'max': 1,
                            'min': 0.8
                        },
                        'patch_size': {'values': [2,4,8,16]},
                        'embed_dim': {'values': [16,32,64]},
                        'num_encoder_layers': {'values': [1,2,3,4,5]},
                        'number_hidden_layers': {'values': [1,2,3]},
                        'hidden_layer_depth': {'values': [64,128,256,512,1024]},
                        'head_dim': {'values': [8,16,32,64,128]},
                        'num_heads': {'values': [1,2,3,4,5,6,7,8,9,10]},
                        'dropout':{'values': [0,0.25,0.5,0.75]},
                        'mlp_head_number_hidden_layers': {'values': [1,2,3,4,5]},
                        'mlp_head_hidden_layers_depth': {'values': [64,128,256,512, 1024]},
                        'batch_size': {'values': [128,256]},
                        #'augmentation_ratio': {'values': [0,0.25,0.5,0.75,1.0]},
                        'weight_decay':{'values': [0,0.0001,0.001,0.01,0.1]},
    }

    # Create and run the sweep
    logger.set_sweep_config(metric_name="Validation Accuracy", metric_goal="maximize", hyperparameters=hyperparameters)
    logger.create_sweep()
    logger.set_training_function(tune)
    logger.run_sweep(count=50)

    # Retrieve the best hyperparameters
    api = wandb.Api()
    runs = api.runs("dlvc_group_13/vit_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    train(best_hyperparameters)