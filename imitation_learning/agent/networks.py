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
            nn.Conv2d(in_channels=history_length + 1, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
	        nn.Linear(in_features=1600, out_features=out_features)
	    )


    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

