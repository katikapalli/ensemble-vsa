import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchinfo import summary

class InceptionV3(nn.Module):
    def __init__(self, num_classes=7):
        super(InceptionV3, self).__init__()

        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.AuxLogits = nn.Identity()

        self.base_model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.base_model(x)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        return logits
