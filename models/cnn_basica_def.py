# models/cnn_basica_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # "f" probablemente era tu bloque de features
        self.f = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # "c" tu bloque clasificador
        self.c = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.f(x)
        x = self.c(x)
        return x
