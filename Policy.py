import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_cells=5):
        super(PolicyNetwork, self).__init__()
        self.num_cells = num_cells
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 13 * 13, 256)
        self.fc_mean = nn.Linear(256, num_cells * 2)  # Mean of the action distribution
        self.fc_std = nn.Linear(256, num_cells * 2)   # Standard deviation of the action distribution

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = x.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.reshape(-1, 64 * 13 * 13)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-5  # Ensure std is positive
        return mean.view(-1, self.num_cells, 2), std.view(-1, self.num_cells, 2)
