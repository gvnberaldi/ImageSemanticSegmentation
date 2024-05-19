import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import v2
from tqdm import tqdm

from assignments.assignment_2.dlvc.dataset.oxfordpets import OxfordPetsCustom
from assignments.assignment_2.dlvc.models.segformer import SegFormer
from assignments.assignment_2.dlvc.models.segment_model import DeepSegmenter

def get_test_data(dataset_path):
    test_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform2 = v2.Compose([v2.ToImage(),
                                   v2.ToDtype(torch.long, scale=False),
                                   v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])  # ,

    dataset = OxfordPetsCustom(root=dataset_path,
                               split='test',
                               download=True,
                               target_types='segmentation',
                               transform=test_transform,
                               target_transform=test_transform2)
    return dataset


def plot_confusion_matrix(data_loader, model, model_name, output_dir):
    all_labels = []
    all_preds = []

    # Iterate over the dataset
    for images, masks in tqdm(data_loader, total=len(data_loader), desc='Testing images'):
        with torch.no_grad():
            outputs = model(images)

            if isinstance(outputs, dict):
                outputs = outputs['out']
            preds = torch.argmax(outputs, dim=1)
            # Flatten the tensors
            masks = masks.view(-1).cpu().numpy()
            preds = preds.view(-1).cpu().numpy()
            masks = masks - 1

            all_labels.extend(masks)
            all_preds.extend(preds)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Extract class names from the classes_seg attribute
    class_names = [label.name for label in OxfordPetsCustom.classes_seg]

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(output_dir)
    plt.show()

    return all_labels, all_preds


def plot_roc_curve(all_labels, all_preds, num_classes, model_name, output_dir):
    # Compute ROC curve and ROC area for each class
    # Binarize the output
    labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    preds_binarized = label_binarize(all_preds, classes=range(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], preds_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Extract class names from the classes_seg attribute
    class_names = [label.name for label in OxfordPetsCustom.classes_seg]

    # Plot ROC curve for each class
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir)
    plt.show()


if __name__ == "__main__":
    test_data = get_test_data(os.path.join(os.path.dirname(__file__), 'data\\oxfordpets'))
    num_classes = len(test_data.classes_seg)
    # Create DataLoader
    data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate FCN ResNet-50 model
    model = DeepSegmenter(fcn_resnet50(weights=None, num_classes=num_classes))
    model.to(device)
    print('Loading weights...')
    model.load_all_weights(path='C:\\Users\\Utente\\Desktop\\FCN_model_best.pth')
    model.eval()

    all_labels, all_preds = plot_confusion_matrix(data_loader, model,
                                                  'FCN ResNet-50',
                                                  'C:\\Users\\Utente\\Desktop\\FCN_confusion_matrix.png')

    plot_roc_curve(all_labels, all_preds, num_classes,
                   'FCN ResNet-50',
                   'C:\\Users\\Utente\\Desktop\\FCN_roc_curve.png')

    # Instantiate SegFormer model
    model = DeepSegmenter(SegFormer(num_classes=num_classes))
    model.to(device)
    print('Loading weights...')
    model.load_all_weights(path='C:\\Users\\Utente\\Desktop\\SegFormer_model_best.pth')
    model.eval()

    all_labels, all_preds = plot_confusion_matrix(data_loader, model,
                                                  'SegFormer',
                                                  'C:\\Users\\Utente\\Desktop\\SegFormer_confusion_matrix.png')

    plot_roc_curve(all_labels, all_preds, num_classes,
                   'SegFormer',
                   'C:\\Users\\Utente\\Desktop\\SegFormer_roc_curve.png')
