import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
  def __init__(self):
    super(BaselineCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(32 * 64 * 64, 1)


  def forward(self, x):
    # Apply two pooling layers
    x = self.pool1(
      torch.relu(
        self.conv1(x)  # Pass the input through the first convolutional layer
      )  # Apply the ReLU activation function to the output of the first convolutional layer
    )  # Apply the first pooling layer
    x = self.pool2(
      torch.relu(
        self.conv2(x)  # Pass the output of the first convolutional layer through the second convolutional layer}
      )  # Apply the ReLU activation function to the output of the second convolutional layer
    )  # Apply the second pooling layer

    x = x.view(-1, 32 * 64 * 64)  # Flatten after pooling

    x = self.fc(
      torch.relu(x)
    )  # Pass the flattened input through the fully connected layer
    return x
