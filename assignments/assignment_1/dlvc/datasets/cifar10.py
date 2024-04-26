import os
from typing import Tuple
import numpy as np
import pickle
import random

from dlvc.datasets.dataset import Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    _datasets = {}

    # Singleton class
    def __new__(cls, subset: Subset, *args, **kwargs):
        if subset in cls._datasets:
            return cls._datasets[subset]

        instance = super(CIFAR10Dataset, cls).__new__(cls)
        cls._datasets[subset] = instance
        return instance

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''
        self.fdir = os.path.join(os.getcwd(), fdir)  # Base dataset folder directory

        # Check if the path is a directory
        #if not os.path.isdir(self.fdir):
        #    print(self.fdir)
        #    raise ValueError("The provided path is not a directory")

        # List of file names expected in the CIFAR-10 folder
        expected_files = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

        # Check if all necessary files are present
        for expected_file in expected_files:
            if not os.path.isfile(os.path.join(self.fdir, expected_file)):
                raise ValueError(f"File '{expected_file}' is missing")

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = transform
        self.augmentation_transform = None
        self.augment_probability = None
        self.subset = subset
        self.images, self.labels = self._load_data()

    def set_augmentation_transform(self, augmentation_transform=None, augment_probability=0.5):
        self.augmentation_transform = augmentation_transform
        self.augment_probability = augment_probability

    def set_augmentation_probability(self, augment_probability=0.5):
        self.augment_probability = augment_probability

    def _load_data(self):
        if self.subset == Subset.TRAINING:
            return self._load_train_data()
        elif self.subset == Subset.VALIDATION:
            return self._load_validation_data()
        elif self.subset == Subset.TEST:
            return self._load_test_data()
        else:
            raise ValueError("Invalid subset value")

    def _unpickle(self, file_path):
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _load_cifar10(self, file_path):
        data = self._unpickle(os.path.join(self.fdir, file_path))
        images = data[b'data']
        labels = data[b'labels']
        images = images.reshape(-1, 3, 32, 32)  # Reshape images to (num_samples, channels, height, width)
        images = images.transpose(0, 2, 3, 1)   # Transpose to (num_samples, height, width, channels)
        return images, labels

    def _load_train_data(self):
        train_images = []
        train_labels = []
        for i in range(1, 5):
            batch_images, batch_labels = self._load_cifar10(f'data_batch_{i}')
            train_images.append(batch_images)
            train_labels.extend(batch_labels)

        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.array(train_labels)
        return train_images, train_labels

    def _load_validation_data(self):
        return self._load_cifar10('data_batch_5')

    def _load_test_data(self):
        return self._load_cifar10('test_batch')

    def __len__(self) -> int:
        # Returns the number of samples in the dataset.
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        if idx < 0 or idx >= len(self.images):
            raise IndexError("Index out of bounds")
        image, label = self.images[idx], self.labels[idx]
        image = self.transform(image)
        return image, label


    def num_classes(self) -> int:
        # Returns the number of classes.
        return len(np.unique(self.labels))
