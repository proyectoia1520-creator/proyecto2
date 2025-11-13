import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Bloque de características (features)
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Bloque de clasificación (classifier)
        self.c = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.f(x)
        x = self.c(x)
        return x
