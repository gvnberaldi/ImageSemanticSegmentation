from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes
        self.correct_predictions = 0
        self.total_predictions = 0
        self.class_correct_predictions = [0] * len(self.classes)
        self.class_total_predictions = [0] * len(self.classes)

        self.per_class_accuracies = np.zeros(len(self.classes))
        self.overall_accuracy = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.correct_predictions = 0
        self.total_predictions = 0
        self.class_correct_predictions = [0] * len(self.classes)
        self.class_total_predictions = [0] * len(self.classes)

        self.per_class_accuracies = np.zeros(len(self.classes))
        self.overall_accuracy = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Number of classes in prediction ({prediction.shape[1]}) \
                               does not match the expected number of classes ({len(self.classes)}).")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Number of samples in prediction and target tensors do not match.")
        if target.min().item() < 0 or target.max().item() >= len(self.classes):
            raise ValueError("Target values are out of bounds.")

        predicted_classes = torch.argmax(prediction, dim=1)
        self.correct_predictions += (predicted_classes == target).sum().item()
        self.total_predictions += target.size(0)

        for i in range(len(self.classes)):
            class_mask = (target == i)  # Index of samples that belong to class i
            self.class_correct_predictions[i] += (predicted_classes[class_mask] == i).sum().item()
            self.class_total_predictions[i] += class_mask.sum().item()

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''
        performance_str = f'Overall Accuracy: {self.accuracy():.4f}\n'
        performance_str += f'Per Class Accuracies:{self.per_class_accuracy():.4f} \n'

        for i in range(len(self.classes)):
            performance_str += f'Accuracy for class {self.classes[i]} is: {self.per_class_accuracies[i]:.4f} \n'

        return performance_str

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        if self.total_predictions == 0:
            return 0.0
        self.overall_accuracy = self.correct_predictions / self.total_predictions
        return self.overall_accuracy

    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        if self.total_predictions == 0:
            return 0.0
        for i in range(len(self.classes)):
            if self.class_total_predictions[i] != 0:
                self.per_class_accuracies[i] = self.class_correct_predictions[i] / self.class_total_predictions[i]
        return self.per_class_accuracies.mean()
       