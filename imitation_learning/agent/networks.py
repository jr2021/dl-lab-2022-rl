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
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=history_length + 1, out_channels=16, kernel_size=5, stride=4),
            nn.ReLU(),
#	    nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
#	    nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_features)
        )

        self.model.cuda()


    def forward(self, x):
        # TODO: compute forward pass
        x = self.model(x)

        return x

