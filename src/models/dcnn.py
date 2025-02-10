import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ProposedDCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ProposedDCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
        self.bn6 = nn.BatchNorm2d(256)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)
        self.drop3 = nn.Dropout(p=0.5)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

        self.fc1 = nn.Linear(4096, 128)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.bn7 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(p=0.6)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop1(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop2(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)
        return x