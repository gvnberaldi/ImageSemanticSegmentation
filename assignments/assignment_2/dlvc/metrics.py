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
        self.confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.confusion_matrix = torch.zeros(len(self.classes), len(self.classes))

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

        # Iterate over each pixel updating the confusion matrix
        for i in range(prediction.shape[0]):  # iterate over batch size
            for j in range(prediction.shape[2]):  # iterate over height
                for k in range(prediction.shape[3]):  # iterate over width
                    pred_pixel = prediction[i, :, j, k]  # prediction for each class
                    true_label = int(target[i, j, k])  # ground truth label
                    if true_label != 255:  # ignore pixel with value 255
                        pred_label = torch.argmax(pred_pixel)  # predicted class for pixel (j,k)
                        self.confusion_matrix[true_label, pred_label] += 1

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIou: {self.mIoU():.2f}"
    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection
        iou = intersection / union
        iou[torch.isnan(iou)] = 0  # Handle division by zero
        return iou.mean().item()





