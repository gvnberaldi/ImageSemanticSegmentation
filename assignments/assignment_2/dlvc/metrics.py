import time
from abc import ABCMeta, abstractmethod
import torch

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


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.intersection = torch.zeros(len(self.classes), dtype=torch.float32)
        self.union = torch.zeros(len(self.classes), dtype=torch.float32)
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.intersection = torch.zeros(len(self.classes), dtype=torch.float32)
        self.union = torch.zeros(len(self.classes), dtype=torch.float32)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored.
        '''

        # Check if prediction and target have compatible shapes
        if (prediction.shape[0] != target.shape[0] or
            prediction.shape[2:] != target.shape[1:] or
            prediction.shape[1] != len(self.classes)):
            raise ValueError(
                "Prediction and target shapes do not match or are incompatible with the number of classes.")

        # Get the predicted classes by taking the argmax over the class dimension
        # Shape of pred_classes: (s, h, w)
        pred_classes = prediction.argmax(dim=1)
        # valid_mask: Valid pixels (where target is not equal to 255)
        valid_mask = target != 255

        # Iterate over each class to calculate intersection and union per class
        for cls in range(len(self.classes)):
            # Create binary masks for the current class
            # pred_mask: Pixels predicted as the current class
            # true_mask: Pixels labeled as the current class
            pred_mask = (pred_classes == cls) & valid_mask
            true_mask = (target == cls) & valid_mask

            # Calculate the intersection and union for the current class
            # Intersection: Pixels where both masks are True (True Positive)
            # Union: Pixels where at least one mask is True (True Positive + False Positive + False Negative)
            self.intersection[cls] += torch.sum(pred_mask & true_mask).item()
            self.union[cls] += torch.sum(pred_mask | true_mask).item()

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIou: {self.mIoU():.2f}"

    def mIoU(self):
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''

        # Calculate IoU for each class
        iou = self.intersection / self.union
        iou[torch.isnan(iou)] = 0  # Handle division by zero
        # Compute the mean IoU across all classes
        return iou.mean().item()





