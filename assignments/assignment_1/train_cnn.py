## Feel free to change the imports according to your implementation and needs
import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchinfo import summary

from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.models.cnn import YourCNN
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset


def train(args):

    transform = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    fdir = "data\\cifar-10-batches-py"

    train_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TRAINING, transform=transform)
    val_data = CIFAR10Dataset(fdir=fdir, subset=Subset.VALIDATION, transform=transform)
    test_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TEST, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DeepClassifier(YourCNN())
    model = model.to(device)
    summary(model, input_size=(128,3, 32, 32))

    optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models\\cnn")
    model_save_dir.mkdir(exist_ok=True)

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
                                       args.num_epochs,
                                       model_save_dir,
                                       batch_size=128,  # feel free to change
                                       val_frequency=val_frequency)
    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    if not isinstance(args, tuple):
        args = args.parse_args()

    args.num_epochs = 30
    train(args)
