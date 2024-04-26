import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm, trange

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5,
                 logger=None) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.logger = logger

        # Create data loaders
        self.training_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """

        # Logic mostly from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        running_loss = 0.
        self.train_metric.reset()

        for i, data in enumerate(self.training_loader):
            # print(self.model.device)
            # Every data instance is an input + label pair
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.long().to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            self.outputs = self.model(inputs)
            # Compute the loss and its gradients
            self.loss = self.loss_fn(self.outputs, labels)
            self.loss.backward()
            # Adjust learning weights
            self.optimizer.step()

            running_loss += self.loss.item()
            self.train_metric.update(prediction=self.outputs, target=labels)

        print(f"Training metric for epoch: {epoch_idx} \n")
        print(f'Loss = {running_loss/(i+1)}')
        print(self.train_metric)

        self.logger.log({'Train Loss': running_loss/(i+1), 'Train Accuracy': self.train_metric.accuracy(), 'Train Per Class Accuracy': self.train_metric.per_class_accuracy()}, step=epoch_idx, commit=False)

        return running_loss/(i+1),  self.train_metric.accuracy(), self.train_metric.per_class_accuracy()

    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        # Logic mostly from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        running_loss = 0.
        self.val_metric.reset()

        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.long().to(self.device)

                # Make predictions for this batch
                self.outputs = self.model(inputs)
                # Compute the loss
                self.loss = self.loss_fn(self.outputs, labels)
                running_loss += self.loss.item()

                # Get class accuracy
                self.val_metric.update(prediction=self.outputs, target=labels)

        print(f"Validation metric for epoch: {epoch_idx} \n")
        print(f'Loss = {running_loss / (i + 1)}')
        print(self.val_metric)

        self.logger.log({'Validation Loss': running_loss/(i+1), 'Validation Accuracy': self.val_metric.accuracy(), 'Validation Per Class Accuracy': self.val_metric.per_class_accuracy()}, step=epoch_idx, commit=False)

        return running_loss/(i+1), self.val_metric.accuracy(), self.val_metric.per_class_accuracy()

    def train(self, save = False) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_accuracy = 0.

        for epoch in trange(self.num_epochs):
            self.model.train()
            self._train_epoch(epoch)
            if (epoch+1) % self.val_frequency == 0:
                self.model.eval()
                val_metrics = self._val_epoch(epoch)
                if val_metrics[1] > best_accuracy and save:
                    best_accuracy = val_metrics[1]
                    self.model.save(save_dir=self.training_save_dir, suffix=f'model.pth')




            
            


