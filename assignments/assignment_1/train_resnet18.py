## Feel free to change the imports according to your implementation and needs
import argparse
import torch
from pathlib import Path
import os

from dlvc.models.class_model import DeepClassifier # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.wandb_logger import WandBLogger
from dlvc.utils import get_datasets, get_api_key

from torchvision.models import resnet18
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary


def train(args):

    train_data, val_data, _ = get_datasets()
    
    resnet = resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DeepClassifier(resnet)
    model = model.to(device)
    summary(model, input_size=(128, 3, 32, 32))

    optimizer = AdamW(model.parameters(),lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models\\resnet")

    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    logger = WandBLogger(api_key=get_api_key(), model=model, project_name="training", entity_name="dlvc_group_13", run_name="ResNet")
    
    trainer = ImgClassificationTrainer( model,
                                        optimizer,
                                        loss_fn,
                                        lr_scheduler,
                                        train_metric,
                                        val_metric,
                                        train_data,
                                        val_data,
                                        device,
                                        args.num_epochs,
                                        model_save_dir,
                                        batch_size=128,
                                        val_frequency=val_frequency,
                                        logger=logger)
    trainer.train(save=True)


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0 
    args.num_epochs = 30

    train(args)