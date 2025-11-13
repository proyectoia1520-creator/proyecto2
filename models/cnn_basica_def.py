# models/cnn_basica_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.f = nn.Sequential(
            # f.0
            nn.Conv2d(3, 32, 3, padding=1),
            # f.1
            nn.ReLU(),
            # f.2
            nn.Conv2d(32, 64, 3, padding=1),
            # f.3
            nn.ReLU(),
            # f.4
            nn.Conv2d(64, 128, 3, padding=1),
            # f.5
            nn.ReLU(),
            # f.6
            nn.Conv2d(128, 256, 3, padding=1),
            # f.7
            nn.ReLU(),
            # f.8
            nn.AdaptiveAvgPool2d((1, 1)),
            # f.9
            nn.Flatten()
        )

        # c.0
        self.c = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.f(x)
        x = self.c(x)
        return x
