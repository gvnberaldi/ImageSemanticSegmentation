## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dlvc.metrics import Accuracy
from dlvc.utils import get_datasets, get_cnn_model, get_api_key

import wandb


def test(args):
    wandb.login(key=get_api_key())

    # Retrieve the best hyperparameters
    api = wandb.Api()
    runs = api.runs("dlvc_group_13/cnn_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    _, _, test_data = get_datasets()
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=best_hyperparameters['batch_size'],
                                                   shuffle=False)

    model, device = get_cnn_model(best_hyperparameters, os.path.join(os.getcwd(), 'saved_models\\cnn\\model.pth'))
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    total_loss = 0.0
    prediction_list = []
    all_labels = []

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            test_metric.update(prediction=outputs, target=labels)
            # Collect predictions and labels
            prediction_list.append(outputs)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)

    # Calculate test accuracy
    test_accuracy = test_metric.accuracy()
    test_per_class_accuracy = test_metric.per_class_accuracy()
    print("Test Accuracy:", test_accuracy)
    print("Test Per Class Accuracy:", test_per_class_accuracy)

    # Calculate average loss over the test dataset
    average_loss = total_loss / len(test_data_loader)
    print("Test Loss:", average_loss)

    prob_prediction = torch.cat(prediction_list, dim=0)
    _, int_prediction = torch.max(prob_prediction, dim=1)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, int_prediction)
    # Visualize confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.classes,
                yticklabels=test_data.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("CNN Confusion Matrix")
    plt.savefig(os.path.join(os.getcwd(), 'img\\cnn_confusion_matrix.png'), format='png', dpi=700)
    plt.show()

    # Generate ROC curves and compute AUC for each class
    fpr = dict()  # False Positive Rate
    tpr = dict()  # True Positive Rate
    roc_auc = dict()  # AUC scores

    for i in range(len(test_data.classes)):
        # Create binary true labels for the "one-vs-rest" approach
        y_true_binary = (all_labels == i).astype(int)
        # Compute ROC curve and AUC score
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, prob_prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(len(test_data.classes)):
        plt.plot(fpr[i], tpr[i], label=f'Class {test_data.classes[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line indicating random chance
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(os.getcwd(), 'img\\cnn_roc_curve.png'), format='png', dpi=700)
    plt.show()


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='5', type=str,
                      help='index of which GPU to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    test(args)