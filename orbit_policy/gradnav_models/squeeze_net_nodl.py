"""VisualPerceptionNet from upstream models/squeeze_net.py, rewritten without torchvision.

Upstream: `torchvision.models.squeezenet1_0(pretrained=True)`, then the first layer replaced by
`Conv2d(C, 96, 7, stride 2, padding 3, bias=False)` (randomly initialised, frozen).
Here SqueezeNet 1.0 is rebuilt layer by layer from torchvision 0.20's source with the same module names
(squeezenet.features.N, squeezenet.classifier.N, pool, fc), so saved state_dicts load with strict=True.
The classifier is not used in forward; it is kept only so the parameter names match.
"""
import torch
import torch.nn as nn


class Fire(nn.Module):  # torchvision.models.squeezenet.Fire, identical
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                          self.expand3x3_activation(self.expand3x3(x))], 1)


class _SqueezeNet10(nn.Module):  # torchvision SqueezeNet(version="1_0"), first layer replaced as upstream does
    def __init__(self, input_channels=3, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Conv2d(512, num_classes, kernel_size=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))


class VisualPerceptionNet(nn.Module):  # forward identical to upstream
    def __init__(self, input_channels=3, visual_feature_size=24):
        super().__init__()
        self.squeezenet = _SqueezeNet10(input_channels)
        for param in self.squeezenet.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, visual_feature_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.squeezenet.features(x)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
