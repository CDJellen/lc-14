import torch.nn as nn
import torch.nn.functional as F


class FinalModel(nn.Module):
  def __init__(self):
    super(FinalModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16) 
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(32) 
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(32 * 64 * 64, 1)

  def forward(self, x):
    x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Conv1 → BatchNorm → ReLU → Pool
    x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Conv2 → BatchNorm → ReLU → Pool

    x = x.view(-1, 32 * 64 * 64)  # Flatten for fully connected layer
    x = self.fc(x)  # Linear layer (output)

    return x
    