import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Bloque de características: coincide con f.0, f.3, f.6, f.9
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # f.0
            nn.ReLU(),                                   # f.1
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # f.3
            nn.ReLU(),                                   # f.4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# f.6
            nn.ReLU(),                                   # f.7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# f.9
            nn.ReLU(),                                   # f.10
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Bloque clasificador: c.3.weight, c.3.bias
        self.c = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.f(x)
        x = self.c(x)
        return x
