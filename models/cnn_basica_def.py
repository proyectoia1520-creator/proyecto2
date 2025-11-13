# models/cnn_basica_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.f0 = nn.Conv2d(3, 32, 3, padding=1)
        self.f3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f6 = nn.Conv2d(64, 128, 3, padding=1)
        self.f9 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.c3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.f0(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f6(x))
        x = F.relu(self.f9(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.c3(x)
        return x
