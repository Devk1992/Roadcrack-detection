from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GaborConv2dLayer import GaborConv2dLayer

class GaborNN5layer(nn.Module):
    def __init__(self):
        super(GaborNN5layer, self).__init__()
        self.g1 =GaborConv2dLayer(3, 32, kernel_size=(15, 15), stride=1)
        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.c3 = nn.Conv2d(128, 256, kernel_size=(1, 1))
        self.c4 = nn.Conv2d(256, 512, kernel_size=(1, 1))
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c3(x)),kernel_size=1)
        x = F.max_pool2d(F.leaky_relu(self.c4(x)), kernel_size=1)
        x = nn.Dropout2d()(x)
        x = x.view(-1, 512 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError
