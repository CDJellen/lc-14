import torch.nn as nn


class AlphaModel(nn.Module):
  def __init__(self):
    super(AlphaModel, self).__init__()
    # First convolution layer
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Second convolution layer
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Third convolution layer with more filters
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Fourth convolution layer (new addition)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.relu4 = nn.ReLU()
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Fully connected layers
    self.fc1 = nn.Linear(128 * 32 * 32, 256)  # Adjust for the feature map size after pooling
    self.relu5 = nn.ReLU()
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)  # Final output layer

  def forward(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = self.pool3(self.bn3(self.conv3(x)))  # Apply batch normalization after 3rd convolution
    x = self.pool4(self.relu4(self.conv4(x)))  # Apply 4th convolution layer and pooling

    x = x.view(x.size(0), -1)  # Flatten dynamically for batch size
    x = self.relu5(self.fc1(x))  # Fully connected layer
    x = self.fc2(x)  # Second fully connected layer
    x = self.fc3(x)  # Final prediction
    return x
