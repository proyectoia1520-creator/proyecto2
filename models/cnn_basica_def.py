import torch
import torch.nn as nn
from collections import OrderedDict

class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Definir con nombres explícitos iguales a los del checkpoint
        self.f = nn.Sequential(OrderedDict([
            ("0", nn.Conv2d(3, 32, 3, padding=1)),
            ("1", nn.ReLU()),
            ("3", nn.Conv2d(32, 64, 3, padding=1)),
            ("4", nn.ReLU()),
            ("6", nn.Conv2d(64, 128, 3, padding=1)),
            ("7", nn.ReLU()),
            ("9", nn.Conv2d(128, 256, 3, padding=1)),
            ("10", nn.ReLU()),
            ("11", nn.AdaptiveAvgPool2d((1, 1))),
            ("12", nn.Flatten())
        ]))

        self.c = nn.Sequential(OrderedDict([
            ("3", nn.Linear(256, num_classes))
        ]))

    def forward(self, x):
        x = self.f(x)
        x = self.c(x)
        return x
