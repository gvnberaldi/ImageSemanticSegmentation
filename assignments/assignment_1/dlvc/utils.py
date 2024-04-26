import os

import torch
import torchvision.transforms.v2 as v2
from torchinfo import summary

from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.models.cnn import YourCNN

GVN_API_KEY = 'e5e7b3c0c3fbc088d165766e575853c01d6cb305'


def get_api_key():
    return GVN_API_KEY


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
    test_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TEST, transform=transform)

    train_data.set_augmentation_transform(augmentation_transform=augmentation_transform)

    return train_data, val_data, test_data


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