import torch
import torch.nn as nn
from typing import Optional

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock3d(nn.Module):
    """Basic Block for ResNet-18/34"""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class NeuroSpectrumNet(nn.Module):
    """
    3D ResNet-18 with Multi-Head Classification.
    
    Heads:
    1. GenderHead (4 classes: ASD-M, ASD-F, TD-M, TD-F)
    2. AgeHead (4 classes: ASD-Child, ASD-Adult, TD-Child, TD-Adult)
    3. OctalHead (8 classes: Combined)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = 64
        
        # Initial Conv
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification Heads
        # Benchmark papers typically use 512 feature vectors
        self.gender_head = nn.Linear(512, 4)
        self.age_head = nn.Linear(512, 4)
        self.octal_head = nn.Linear(512, 8)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock3d(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * ResidualBlock3d.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Backbone Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1) # (B, 512)

        # Multi-Head Output
        # Return dictionary for flexibility
        return {
            'gender_logits': self.gender_head(features),
            'age_logits': self.age_head(features),
            'octal_logits': self.octal_head(features),
            'features': features # Useful for Grad-CAM
        }

if __name__ == "__main__":
    # Smoke Test
    model = NeuroSpectrumNet()
    dummy = torch.randn(2, 1, 64, 64, 64) # Batch 2, 64^3 volume
    out = model(dummy)
    print("Model Output Keys:", out.keys())
    print("Octal Shape:", out['octal_logits'].shape)
