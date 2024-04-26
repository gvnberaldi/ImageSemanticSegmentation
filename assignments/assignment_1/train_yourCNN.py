## Feel free to change the imports according to your implementation and needs
from pathlib import Path
import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.utils import get_cnn_model, get_datasets, get_api_key
from dlvc.wandb_logger import WandBLogger
from dlvc.wandb_sweep import WandBHyperparameterTuning

LOGGER = None


def tune():
    with wandb.init() as run:
        config = run.config

        model, device = get_cnn_model(config)
        model_save_dir = Path("saved_models\\cnn")

        train_data, val_data, _ = get_datasets()
        #train_data.set_augmentation_probability(augment_probability=config['augmentation_ratio'])
        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

        optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
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
    model, device = get_cnn_model(best_hyperparameters)
    model_save_dir = Path("saved_models\\cnn")

    train_data, val_data, _ = get_datasets()
    #train_data.set_augmentation_probability(augment_probability=best_hyperparameters['augmentation_ratio'])
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    logger = WandBLogger(api_key=get_api_key(), model=model, project_name="training", entity_name="dlvc_group_13", run_name="CNN")

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
    trainer.train(save=True)


if __name__ == "__main__":

    logger = WandBHyperparameterTuning(api_key=get_api_key(), project_name="cnn_tuning", entity_name="dlvc_group_13")
    LOGGER = logger

    hyperparameters = {
        'batch_size': [128, 256, 512],
        'dropout_rate': [0, 0.3, 0.5, 0.7],
        'conv_layer_1_dim': [16, 32],
        'conv_layer_2_dim': [32, 64],
        'conv_layer_3_dim': [64, 128],
        'mlp_layer_1_dim': [128, 256],
        'mlp_layer_2_dim': [62, 128],
        # 'augmentation_ratio': {
        #    'distribution': 'uniform',
        #    'max': 1,
        #    'min': 0
        #}
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
