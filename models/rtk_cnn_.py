import torch
from torch import nn
import torch.nn.functional as F
from avalanche.models import BaseModel


class RtkModel(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(50688, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class RtkModel(nn.Sequential):
#     def __init__(self):
#         super().__init__(
#             nn.Conv2d(3, 32, 3, stride=1, padding=1),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=1, padding=1),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(50688, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)
#         )
