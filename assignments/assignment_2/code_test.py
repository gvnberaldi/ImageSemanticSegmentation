import os

import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.dataset.cityscapes import CityscapesCustom


def denormalize_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Denormalize the image by multiplying by the standard deviation and adding back the mean
    denormalized_image = (image * std) + mean
    # Clip the values to ensure they are within the valid range [0, 1]
    denormalized_image = np.clip(denormalized_image, 0, 1)

    return denormalized_image


def plot_images_and_labels(dataloader):
    # Get a batch of data
    images, labels = next(iter(dataloader))

    # Plot the images with their respective labels
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert tensors to numpy array
        image = denormalize_image(image.numpy().transpose(1, 2, 0))
        label = label.numpy().astype(np.uint8).transpose(1, 2, 0)
        label[label == 255] = 19

        # Plot the image
        axes[0, i].imshow(image)
        axes[0, i].axis('off')
        axes[0, i].set_title('Image')

        # Plot the label
        axes[1, i].imshow(label, cmap='tab20b')
        axes[1, i].axis('off')
        axes[1, i].set_title('Label')

    plt.tight_layout()
    plt.show()


def datasets_test():
    image_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    label_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.long, scale=False),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST)])  # ,

    # OxfordPets dataset
    print('TESTING OXFORDPETS DATASET')
    dataset_path = os.path.join(os.path.dirname(__file__), 'data\\oxfordpets')
    download = False if os.path.exists(os.path.join(dataset_path, 'oxford-iiit-pet')) else True

    train_data = OxfordPetsCustom(root=dataset_path,
                                  split="trainval",
                                  target_types='segmentation',
                                  transform=image_transform,
                                  target_transform=label_transform,
                                  download=download)

    val_data = OxfordPetsCustom(root=dataset_path,
                                split="test",
                                target_types='segmentation',
                                transform=image_transform,
                                target_transform=label_transform,
                                download=download)

    # print(train_data)
    # print(val_data)

    # Create a DataLoader with batch size 4
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    plot_images_and_labels(train_loader)

    # Cityscapes dataset
    print('TESTING CITYSCAPES DATASET')
    dataset_path = os.path.join(os.path.dirname(__file__), 'data\\cityscapes')
    train_data = CityscapesCustom(root=dataset_path,
                                  split="train",
                                  mode="fine",
                                  target_type='semantic',
                                  transform=image_transform,
                                  target_transform=label_transform)
    val_data = CityscapesCustom(root=dataset_path,
                                split="val",
                                mode="fine",
                                target_type='semantic',
                                transform=image_transform,
                                target_transform=label_transform)

    # print(train_data)
    # print(val_data)

    # Create a DataLoader with batch size 4
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    plot_images_and_labels(train_loader)


def mIoU_test():
    confusion_matrix = torch.tensor([[10, 2, 3],
                                     [1, 15, 5],
                                     [2, 3, 20]])
    intersection = torch.diag(confusion_matrix)
    print(f'Intersection = {intersection}')
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection
    print(f'Union = {union}')
    iou = intersection / union
    print(f'IoU = {iou}')
    iou[torch.isnan(iou)] = 0  # Handle division by zero
    print(f'mIoU = {iou.mean().item():.2f}')


if __name__ == "__main__":
    datasets_test()
    # mIoU_test()


