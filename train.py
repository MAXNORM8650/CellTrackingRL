from env import CellTrackingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CellTrackerPolicy(nn.Module):
    def __init__(self, num_cells=5):
        super(CellTrackerPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(64 * 13 * 13, 256)
        self.fc2 = nn.Linear(256, num_cells * 2)  # Predict x and y for each cell
        
    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 5, 2)  # Output shape: (batch_size, num_cells, 2)

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=5*2):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = CellTrackerPolicy()
        self._features_dim = features_dim

    def forward(self, observations):
        print("Observations shape before permute:", observations.shape)
        # Correct permutation for observations with shape (batch_size, height, width, channels)
        # observations = observations.permute(0, 3, 2, 1)
        print("Observations shape after permute:", observations.shape)
        output = self.cnn(observations)
        return output.view(output.size(0), -1)  # Flatten output to match features_dim

env = CellTrackingEnv(
    image_dir='synthetic_data/images',
    mask_dir='synthetic_data/masks',
    tracking_file='synthetic_data/tracking/tracking_info.txt'
)

# Check the environment
check_env(env, warn=True)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=5*2),
    net_arch=[]  # No additional layers
)

model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("cell_tracking_model")

# Test the model
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
env.close()
