import torch
from torch import nn


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        # first conv layer, extracting 16 features from the grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # first max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # second conv layer, same kernel size, just to 32 channels now.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # flattening to get a logit, we are also flattening the images to 64 by 64, and through the convs it becomes 14x14. 14*14*32=6272
        self.fc1 = nn.Linear(in_features=6272, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
