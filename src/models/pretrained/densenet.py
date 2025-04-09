import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchinfo import summary

class DenseNet201(nn.Module):
    def __init__(self, num_classes=7):
        super(DenseNet201, self).__init__()
        
        self.base_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1920, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)