"""
Neural network model architecture for fraud detection.
"""

import torch
from torch import nn


class FraudDetectionModel(nn.Module):

    def __init__(self, input_size=13, hidden_size_1=128, hidden_size_2=256):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size_2, 1)  
    def forward(self, x):
      
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def get_device():
    """Get the device (CUDA if available, otherwise CPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def initialize_model(input_size=13, hidden_size_1=128, hidden_size_2=256, device=None):
  
    if device is None:
        device = get_device()
    
    model = FraudDetectionModel(input_size, hidden_size_1, hidden_size_2).to(device)
   
    return model, device
