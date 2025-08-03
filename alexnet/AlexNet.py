import torch
import torch.nn as nn
import torch.nn.functional as F

# trained on cifar10 dataset

class AlexNet_modified(nn.Module):
    def __init__ (self, num_classes=10):
        super(AlexNet_modified, self).__init__()
        self.layers = nn.Sequential(
            # Input layer 
            # layer 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=9, alpha=1e-4, beta=0.75, k=2.0),
            # layer 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=9, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # layer 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # layer 4
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_classes),  # Adjusted
        )

        self.param_list = list(self.layers.parameters()) + list(self.classifier.parameters())


    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x
    
    def loss(self, output, target):
        return F.cross_entropy(output, target)

    def get_parameter(self, target_name):
        for param in self.param_list:
            if hasattr(param, 'name') and param.name == target_name:
                return param
        return None

    def num_parameters(self):
        return len(self.param_list)

