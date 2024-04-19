import torch.nn as nn

class YourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Second Convolutional Layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Third Convolutional Layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()  # Flatten layer

        self.classifier = nn.Sequential(
            # First Fully Connected Layer
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            # Second Fully Connected Layer
            nn.Linear(256, 128),
            nn.ReLU(),
            # Dropout Layer
            nn.Dropout(0.5),
            # Output Layer
            nn.Linear(128, 10),
            nn.Softmax(dim=1)  # Softmax activation for class probabilities
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    