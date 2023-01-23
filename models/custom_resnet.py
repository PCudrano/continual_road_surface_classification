import torch
from torch import nn
import torch.nn.functional as F
from avalanche.models import BaseModel


class CustomResNet(nn.Module, BaseModel):
    def __init__(self, resnet):
        super().__init__()
        resnet_feature_layers = list(resnet.children())[:-1]
        resnet_classifier = list(resnet.children())[-1]
        self.features = torch.nn.Sequential(*resnet_feature_layers)
        self.classifier = nn.Linear(in_features=resnet_classifier.in_features, out_features=3)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        return x
