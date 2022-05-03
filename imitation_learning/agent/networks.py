import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, out_features=4): 
        super(CNN, self).__init__()

        # TODO : define layers of a convolutional neural network
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
	    nn.Linear(in_features=3528, out_features=4)
	)


    def forward(self, x):
        # TODO: compute forward pass
        x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

