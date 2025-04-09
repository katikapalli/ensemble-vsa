import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def save(self, checkpoint_dir, filename="model.pth"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.state_dict(), os.path.join(checkpoint_dir, filename))
        print(f"Model saved to {os.path.join(checkpoint_dir, filename)}")

    def load(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")

    def freeze_layers(self, num_layers=None, layer_names=None):
        if num_layers is not None:
            for i, param in enumerate(self.parameters()):
                if i < num_layers:
                    param.requires_grad = False
        elif layer_names is not None:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        else:
            raise ValueError("Either `num_layers` or `layer_names` must be provided.")

    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def summary(self):
        print('base')
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")