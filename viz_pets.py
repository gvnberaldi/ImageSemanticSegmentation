import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import fcn_resnet50

from core.models.segformer import SegFormer
from core.models.segment_model import DeepSegmenter
from train import OxfordPetsCustom


def denormalize_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Denormalize the image by multiplying by the standard deviation and adding back the mean
    denormalized_image = (image * std) + mean
    # Clip the values to ensure they are within the valid range [0, 1]
    denormalized_image = np.clip(denormalized_image, 0, 1)

    return denormalized_image


def plot_images(images, labels, predictions, filename='img/combined.png'):
    fig, axes = plt.subplots(3, len(images), figsize=(12, 9))
    for i in range(len(images)):
        img = denormalize_image(images[i].numpy().transpose(1, 2, 0))
        lbl = labels[i].numpy().astype(np.uint8).transpose(1, 2, 0)
        pred = predictions[i].numpy().astype(np.uint8).transpose(1, 2, 0)

        axes[0, i].imshow(img)
        axes[0, i].set_title("Image")
        axes[0, i].axis('off')

        axes[1, i].imshow(lbl, cmap='tab20b')
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis('off')

        axes[2, i].imshow(pred, cmap='tab20b')
        axes[2, i].set_title("Prediction")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    image_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    label_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.long, scale=False),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST)])  # ,


    dataset_path = os.path.join(os.path.dirname(__file__), 'data\\oxfordpets')
    download = False if os.path.exists(os.path.join(dataset_path, 'oxford-iiit-pet')) else True

    train_data = OxfordPetsCustom(root=dataset_path,
                                  split="trainval",
                                  target_types='segmentation',
                                  transform=image_transform,
                                  target_transform=label_transform,
                                  download=download)

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    images = images.to(device)
    labels = labels.to(device)

    # Instantiate FCN ResNet-50 model
    num_classes = 3
    fcn_resnet_model = DeepSegmenter(fcn_resnet50(weights=None, num_classes=num_classes))
    fcn_resnet_model.to(device)
    print('Loading FCN ResNet weights...')
    fcn_resnet_model.load_all_weights(path='C:\\Users\\Utente\\Desktop\\FCN_model_best.pth')
    fcn_resnet_model.eval()

    # Instantiate SegFormer model
    segformer_model = DeepSegmenter(SegFormer(num_classes=num_classes))
    segformer_model.to(device)
    print('Loading SegFormer weights...')
    segformer_model.load_all_weights(path='C:\\Users\\Utente\\Desktop\\SegFormer_model_best.pth')
    segformer_model.eval()

    # Get predictions
    with torch.no_grad():
        outputs = fcn_resnet_model(images)['out']
        fcn_resnet_predictions = torch.argmax(outputs, dim=1)

    with torch.no_grad():
        outputs = segformer_model(images)
        segformer_predictions = torch.argmax(outputs, dim=1)

    images = images.cpu()
    labels = labels.cpu() - 1
    fcn_resnet_predictions = fcn_resnet_predictions.unsqueeze(1).cpu()
    segformer_predictions = segformer_predictions.unsqueeze(1).cpu()

    plot_images(images, labels, fcn_resnet_predictions, filename="C:\\Users\\Utente\\Desktop\\fcn_resnet_mask.png")
    plot_images(images, labels, segformer_predictions, filename="C:\\Users\\Utente\\Desktop\\segformer_mask.png")



