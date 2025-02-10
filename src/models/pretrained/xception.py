import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from pretrainedmodels import xception
from torchinfo import summary

ssl._create_default_https_context = ssl._create_unverified_context

class Xception(nn.Module):
    def __init__(self, num_classes=7):
        super(Xception, self).__init__()

        self.base_model = xception(pretrained='imagenet')

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.last_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

        for param in self.base_model.last_linear.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)