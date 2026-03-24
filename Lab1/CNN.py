import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, hidden_dim = 128):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 5,
            stride = 1,
            padding = 2
        )   # 输出图象32*28*28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )  # 输出图象32*14*14

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 5,
            stride = 1,
            padding = 2
        )   # 输出图象64*14*14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )   # 输出图象64*7*7
        self.fc = nn.Sequential(
            nn.Flatten(),   # 展平
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)   # 输出为10类
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        output = self.fc(x)

        return output