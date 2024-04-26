## Feel free to change the imports according to your implementation and needs
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchinfo import summary

from dlvc.wandb_sweep import WandBHyperparameterTuning
from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.models.cnn import YourCNN
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.wandb_logger import WandBLogger

import wandb

API_KEY = 'e5e7b3c0c3fbc088d165766e575853c01d6cb305'
LOGGER = None


def get_datasets():
    transform = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Define data augmentation transformations for the training set
    augmentation_transform = v2.Compose([
        v2.ToImage(),
        # Randomly flip the image horizontally
        v2.RandomHorizontalFlip(),
        # Randomly rotate the image
        v2.RandomRotation(15),
        # Randomly crop and resize
        v2.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # Convert the image to a PyTorch tensor
        v2.ToDtype(torch.float32, scale=True),
        # Normalize the image with mean and standard deviation
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    fdir = "data\\cifar-10-batches-py"

    train_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TRAINING, transform=transform)
    val_data = CIFAR10Dataset(fdir=fdir, subset=Subset.VALIDATION, transform=transform)

    train_data.set_augmentation_transform(augmentation_transform=augmentation_transform)

    return train_data, val_data


def get_model(hyperparameter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cnn = YourCNN(conv_layer_1_dim=hyperparameter['conv_layer_1_dim'],
                  conv_layer_2_dim=hyperparameter['conv_layer_2_dim'],
                  conv_layer_3_dim=hyperparameter['conv_layer_3_dim'],
                  mlp_layer_1_dim=hyperparameter['mlp_layer_1_dim'],
                  mlp_layer_2_dim=hyperparameter['mlp_layer_2_dim'],
                  dropout_rate=hyperparameter['dropout_rate'])
    model = DeepClassifier(cnn)
    model = model.to(device)
    summary(model, input_size=(128, 3, 32, 32))

    weight_path = os.path.join(os.getcwd(), 'saved_models\\cnn\\model.pth')
    if os.path.exists(weight_path):
        model.load(weight_path)
        print("Loading model weight...")

    return model, device


def tune():
    with wandb.init() as run:
        config = run.config

        model, device = get_model(config)
        model_save_dir = Path("saved_models\\cnn")

        train_data, val_data = get_datasets()
        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

        optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

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
                                           batch_size=config.batch_size,
                                           val_frequency=val_frequency,
                                           logger=LOGGER)
        trainer.train()


def train(best_hyperparameters):
    model, device = get_model(best_hyperparameters)
    model_save_dir = Path("saved_models\\cnn")

    train_data, val_data = get_datasets()
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    logger = WandBLogger(api_key=API_KEY, model=model, project_name="cnn_training", entity_name="dlvc_group_13", run_name="cnn_training_final")

    trainer = ImgClassificationTrainer(model,
                                       optimizer,
                                       loss_fn,
                                       lr_scheduler,
                                       train_metric,
                                       val_metric,
                                       train_data,
                                       val_data,
                                       device,
                                       120,
                                       model_save_dir,
                                       batch_size=best_hyperparameters['batch_size'],
                                       val_frequency=val_frequency,
                                       logger=logger)
    trainer.train()


if __name__ == "__main__":
    logger = WandBHyperparameterTuning(api_key=API_KEY, project_name="cnn_tuning", entity_name="dlvc_group_13")
    LOGGER = logger

    hyperparameters = {
        'batch_size': [128, 256, 512],  # Different batch sizes for experimentation
        'dropout_rate': [0.3, 0.5, 0.7],  # Different dropout rates for regularization
        'conv_layer_1_dim': [16, 32],
        'conv_layer_2_dim': [32, 64],
        'conv_layer_3_dim': [64, 128],
        'mlp_layer_1_dim': [128, 256],
        'mlp_layer_2_dim': [62, 128],
    }

    # Create and run the sweep
    logger.set_sweep_config(metric_name="Validation Accuracy", metric_goal="maximize", hyperparameters=hyperparameters)
    logger.create_sweep()
    logger.set_training_function(tune)
    logger.run_sweep(count=20)

    # Retrieve the best hyperparameters
    api = wandb.Api()
    runs = api.runs("dlvc_group_13/cnn_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    train(best_hyperparameters)
