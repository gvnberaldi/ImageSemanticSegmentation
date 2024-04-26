## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from dlvc.metrics import Accuracy
from dlvc.utils import get_datasets, get_cnn_model, get_api_key

import wandb


def test(args):
    _, _, test_data = get_datasets()
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    num_test_data = len(test_data)

    wandb.login(key=get_api_key())

    # Retrieve the best hyperparameters
    api = wandb.Api()
    runs = api.runs("dlvc_group_13/cnn_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    model, device = get_cnn_model(best_hyperparameters, os.path.join(os.getcwd(), 'saved_models\\cnn\\model.pth'))
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)

            # Compute accuracy
            test_metric.update(prediction=outputs, target=labels)
            # Collect predictions and labels
            predictions.extend(predicted.cpu().numpy())
            labels.extend(labels.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = test_metric.accuracy()
    test_per_class_accuracy = test_metric.per_class_accuracy()
    print("Test Accuracy:", test_accuracy)
    print("Test Per Class Accuracy:", test_per_class_accuracy)

    # Calculate average loss over the test dataset
    average_loss = total_loss / len(test_data_loader)
    print("Test Loss:", average_loss)

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    # Visualize confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.classes,
                yticklabels=test_data.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
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