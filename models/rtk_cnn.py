import torch
from torch import nn
import torch.nn.functional as F
from avalanche.models import BaseModel


class RtkModel(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(50688, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.features = nn.Sequential(*(
                nn.Conv2d(3, 32, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(45056, 128), # nn.Linear(50688, 128),
                nn.ReLU(),
            ))
        self.classifier = nn.Linear(128, 3)
        # self._input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        return x
