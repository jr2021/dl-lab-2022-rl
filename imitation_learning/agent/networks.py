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
            nn.Conv2d(in_channels=history_length + 1, out_channels=32, kernel_size=5, stride=4),
            nn.ReLU(),
	    nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
	    nn.Dropout(p=0.25)
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
	    nn.Linear(in_features=4096, out_features=256),
	    nn.ReLU(),
	    nn.Linear(in_features=256, out_features=out_features)
	)


    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

