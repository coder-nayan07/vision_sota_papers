import torch
import torch.nn as nn

# Reduced VGG configs: fewer channels per layer to lower parameter count
vgg_cfgs = {
    'VGG11': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG13': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'VGG19': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M',
              256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, vgg_name, num_classes=200, init_weights=True):
        super(VGGNet, self).__init__()

        self.features = self._make_layers(vgg_cfgs[vgg_name])

        # For a 64x64 input, after 5 max pools → 2x2 feature map
        classifier_input_size = 256 * 2 * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def count_parameters(self):
        """Counts the total and trainable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params

