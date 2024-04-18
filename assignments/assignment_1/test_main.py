import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np

from dlvc.datasets.dataset import Subset
from dlvc.datasets.cifar10 import CIFAR10Dataset

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png",np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":

    fdir = "data\\cifar-10-batches-py"
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    training_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TRAINING, transform=transform)
    validation_data = CIFAR10Dataset(fdir=fdir, subset=Subset.VALIDATION)
    test_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TEST)

    print("Training Dataset Size: ", training_data.__len__())
    print("Validation Dataset Size: ", validation_data.__len__())
    print("Test Dataset Size: ", test_data.__len__())

    print('Training Dataset Classes Number: ', training_data.num_classes())
    print('Validation Dataset Classes Number: ', validation_data.num_classes())
    print('Test Dataset Classes Number: ', test_data.num_classes())

    # Aggregate the counts of samples per class across all dataset subsets
    total_samples_per_class = np.bincount(training_data.labels) + \
                              np.bincount(validation_data.labels) + \
                              np.bincount(test_data.labels)

    print("Total Samples Per Class: ", total_samples_per_class)

    # Get a random index
    random_index = np.random.randint(0, test_data.__len__())
    # Retrieve the sample image and label at the random index
    sample_training_image, _ = training_data.__getitem__(random_index)
    sample_test_image, _ = test_data.__getitem__(random_index)

    # Print the shape and type of the random image
    print("Shape of random training (transformed) image:", sample_training_image.shape)
    print("Type of random training image:", sample_training_image.dtype)
    print("Shape of random test (NOT transformed) image:", sample_test_image.shape)
    print("Type of random test image:", sample_test_image.dtype)

    # Print labels of the first 10 training samples
    print("Labels of the first 10 training samples:")
    for i in range(10):
        sample_image, sample_label = training_data.__getitem__(i)
        print(f"Sample {i+1}: {sample_label}")

    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=False, num_workers=2)

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))