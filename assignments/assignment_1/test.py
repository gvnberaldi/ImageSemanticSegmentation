## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
from torchvision.models import resnet18
from torchvision import utils
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dlvc.metrics import Accuracy
from dlvc.models.class_model import DeepClassifier
from dlvc.utils import get_datasets, get_cnn_model, get_api_key

import wandb

#Code from https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def test(args):
    wandb.login(key=get_api_key())
    api = wandb.Api()

    models = {}

    # CNN model
    runs = api.runs("dlvc_group_13/cnn_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    cnn, device = get_cnn_model(best_hyperparameters, os.path.join(os.getcwd(), 'saved_models\\cnn\\model.pth'))
    cnn.to(device)
    cnn.eval()

    '''
    # Instantiate ViT
    runs = api.runs("dlvc_group_13/vit_tuning")
    best_run = max(runs, key=lambda run: run.summary.get("Validation Accuracy", 0))
    best_hyperparameters = best_run.config

    vit, _ = get_vit_model(best_hyperparameters, os.path.join(os.getcwd(), 'saved_models\\vit\\model.pth'))
    vit.to(device)
    vit.eval()
    '''

    # ResNet model
    resnet = resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    resnet = DeepClassifier(resnet)
    resnet.load(os.path.join(os.getcwd(), 'saved_models\\resnet\\model.pth'))
    resnet.to(device)
    resnet.eval()

    models['CNN'] = cnn
    models['ResNet'] = resnet
    # models['ViT'] = vit

    _, _, test_data = get_datasets()
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    #Printing weights
    filter = cnn.net.conv[0].weight.data.clone()
    print(filter.shape)
    visTensor(filter, ch=0, allkernels=False)
    plt.axis('off')
    plt.ioff()
    plt.show()


    for model_name, model in models.items():
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
        print(f"{model_name} Test Accuracy:", test_accuracy)
        print(f"{model_name} Test Per Class Accuracy:", test_per_class_accuracy)

        # Calculate average loss over the test dataset
        average_loss = total_loss / len(test_data_loader)
        print(f"{model_name} Test Loss:", average_loss)

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
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(os.path.join(os.getcwd(), f'img\\{model_name}_confusion_matrix.png'), format='png', dpi=700)
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
        plt.title(f'{model_name} ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(os.getcwd(), f'img\\{model_name}_roc_curve.png'), format='png', dpi=700)
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