import torch.nn as nn


class YourCNN(nn.Module):
    def __init__(self, conv_layer_1_dim, conv_layer_2_dim, conv_layer_3_dim,
                 mlp_layer_1_dim, mlp_layer_2_dim, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(3, conv_layer_1_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_layer_1_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Second Convolutional Layer
            nn.Conv2d(conv_layer_1_dim, conv_layer_2_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_layer_2_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Third Convolutional Layer
            nn.Conv2d(conv_layer_2_dim, conv_layer_3_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_layer_3_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()  # Flatten layer

        self.classifier = nn.Sequential(
            # First Fully Connected Layer
            nn.Linear(conv_layer_3_dim * 4 * 4, mlp_layer_1_dim),
            nn.ReLU(),
            # Second Fully Connected Layer
            nn.Linear(mlp_layer_1_dim, mlp_layer_2_dim),
            nn.ReLU(),
            # Dropout Layer
            nn.Dropout(dropout_rate),
            # Output Layer
            nn.Linear(mlp_layer_2_dim, 10),
            nn.Softmax(dim=1)  # Softmax activation for class probabilities
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    