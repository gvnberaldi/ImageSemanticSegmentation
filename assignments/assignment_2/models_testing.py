import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import v2

from assignments.assignment_2.dlvc.models.segment_model import DeepSegmenter

test_transform = v2.Compose([v2.ToImage(),
                              v2.ToDtype(torch.float32, scale=True),
                              v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_transform2 = v2.Compose([v2.ToImage(),
                               v2.ToDtype(torch.long, scale=False),
                               v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])  # ,


# Load the Oxford Pets dataset for test
num_classes = 3
dataset_path = os.path.join(os.path.dirname(__file__), 'data\\oxfordpets')
dataset = datasets.OxfordIIITPet(root=dataset_path,
                                 split='test',
                                 download=True,
                                 target_types='segmentation',
                                 transform=test_transform,
                                 target_transform=test_transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSegmenter(fcn_resnet50(weights=None, num_classes=num_classes))
model.to(device)
model.load_all_weights(path=None)
model.eval()

all_labels = []
all_preds = []

# Iterate over the dataset
for images, masks in data_loader:
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # Flatten the tensors
        masks = masks.view(-1).cpu().numpy()
        preds = preds.view(-1).cpu().numpy()

        all_labels.extend(masks)
        all_preds.extend(preds)

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Create confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

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

# Plot ROC curve for each class
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
