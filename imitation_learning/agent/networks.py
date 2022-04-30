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
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((6, 6))
        # )

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(in_features=9216, out_features=4)


    def forward(self, x):
        # TODO: compute forward pass
        # x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

