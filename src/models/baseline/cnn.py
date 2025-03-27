import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
  def __init__(self):
    super(BaselineCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(64 * 64 * 32, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x = self.pool1(self.relu(self.conv1(x)))
    x = self.pool2(self.relu(self.conv2(x)))

    x = x.view(-1, 32 * 64 * 64)  # Flatten the tensor

    x = self.fc1(self.relu(x))
    x = self.fc2(self.relu(x))

    return x
  