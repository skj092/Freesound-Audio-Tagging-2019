import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        # Load the model with pretrained weights if specified
        if pretrained:
            self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        else:
            self.model = models.resnet18(weights=None)

        # Modify the first convolution layer for grayscale input (1 channel)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Get the input features of the final fully connected layer
        self.in_features = self.model.fc.in_features

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
