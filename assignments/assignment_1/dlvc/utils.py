import os

import torch
from torch import nn
import torchvision.transforms.v2 as v2
from torchinfo import summary

from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.models.cnn import YourCNN
from dlvc.models.vit import VisionTransformer


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


def get_cnn_model(hyperparameter, weight_path = None):
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

    if weight_path and os.path.exists(weight_path):
        model.load(weight_path)
        print("Loading model weight...")

    return model, device


def get_vit_model(hyperparameter, weight_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vit = VisionTransformer(
            img_size=32,
            patch_size=hyperparameter['patch_size'],
            in_chans=3,
            embed_dim=hyperparameter['embed_dim'],
            num_encoder_layers=hyperparameter['num_encoder_layers'],
            hidden_layer_depth=hyperparameter['hidden_layer_depth'],
            head_dim=hyperparameter['head_dim'],
            num_heads = hyperparameter['num_heads'],
            norm_layer=nn.LayerNorm,
            activation_function=nn.GELU,
            dropout=hyperparameter['dropout'],
            num_classes = 10,
            mlp_head_hidden_layers_depth=hyperparameter['mlp_head_hidden_layers_depth']
        )

    model = DeepClassifier(vit)
    model = model.to(device)
    summary(model, input_size=(hyperparameter['batch_size'], 3, 32, 32))
    if weight_path and os.path.exists(weight_path):
        model.load(weight_path)
        print("Loading model weight...")
    return model, device