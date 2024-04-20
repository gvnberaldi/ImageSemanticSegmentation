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

import wandb

def train(config=None):

    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure 
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation
    
    with wandb.init(config=config):
        config = wandb.config

        #Data 
        train_transform = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
        
        val_transform = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    

        fdir = "data\\cifar-10-batches-py"

        train_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TRAINING, transform=train_transform)
        val_data = CIFAR10Dataset(fdir=fdir, subset=Subset.VALIDATION, transform=val_transform)
        test_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TEST)
    
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Network

        network = VisionTransformer(
            img_size=32,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
            num_encoder_layers=config.num_encoder_layers,
            number_hidden_layers=config.number_hidden_layers,
            hidden_layer_depth = config.hidden_layer_depth,
            head_dim=config.head_dim,
            num_heads = config.num_heads,
            norm_layer=nn.LayerNorm,
            activation_function=nn.GELU,
            dropout=config.dropout,
            num_classes = 10,
            mlp_head_number_hidden_layers=config.mlp_head_number_hidden_layers,
            mlp_head_hidden_layers_depth=config.mlp_head_hidden_layers_depth
        )
        summary(network)

        optimizer = AdamW(network.parameters(),lr=config.lr, amsgrad=True) if config.optimizer == 'AdamW' else SGD(network.parameters(),lr=config.lr, momentum = 0.9)

        loss_fn = torch.nn.CrossEntropyLoss()
    
        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

        model_save_dir = Path("saved_models")
        model_save_dir.mkdir(exist_ok=True)

        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.gamma)

        network = network.to(device)
    
        trainer = ImgClassificationTrainer(network, 
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
                        batch_size=config.batch_size, # feel free to change
                        val_frequency = val_frequency)
        trainer.train()


if __name__ == "__main__":
    # Perform a parameter sweep using the provided functionality from WandB https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=ImLCIMOoIe5Y
    wandb.login(key="5a4726d6cfbe6bf6fa6cdab8143ed9b4f47db04d")
    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'optimizer': {
            'values': ['AdamW', 'sgd']
        },

        'scheduler' : {
            'value': 'ExponentialLR'
        },

        'lr':{
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1 
        }, 

        'gamma':{
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1 
        },

        'patch_size': {
            'values': [2,4,8,16]
        },

        'embed_dim': {
            'values': [4,8,16,32,64]
        },

        'num_encoder_layers': {
            'values': [1,2,3,4,5]
        },

        'number_hidden_layers': {
            'values': [1,2,3,4,5]
        },

        'hidden_layer_depth': {
            'values': [4,8,16,32,64]
        },

        'head_dim': {
            'values': [4,8,16,32,64]
        },

        'num_heads': {
            'values': [1,2,3,4,5]
        },

        'dropout':{
            'distribution': 'uniform',
            'min': 0,
            'max': 0.5 
        },

        'mlp_head_number_hidden_layers': {
            'values': [1,2,3,4,5]
        },

        'mlp_head_hidden_layers_depth': {
            'values': [4,8,16,32,64]
        },

        'batch_size': {
        # integers between 32 and 1024
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 1024,
      }

    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project = 'dlvc_ass_1_vit_sweep')

    wandb.agent(sweep_id, train, count=100)