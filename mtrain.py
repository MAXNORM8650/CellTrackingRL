import torch
import torch.optim as optim
import torch.nn.functional as F
from env import CellTrackingEnv
from Policy import PolicyNetwork
import numpy as np

def train(env, policy_net, optimizer, num_episodes=1000, gamma=0.99):
    policy_net.train()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = policy_net(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob.sum())

            action_np = action.detach().numpy()[0]
            next_state, reward, done, _ = env.step(action_np)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Compute returns and losses
        returns = compute_returns(rewards, gamma)
        loss = compute_loss(log_probs, returns)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Loss: {loss.item():.4f}")

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

def compute_loss(log_probs, returns):
    log_probs = torch.stack(log_probs)
    returns = returns.detach()
    loss = - (log_probs * returns).sum()
    return loss

if __name__ == "__main__":
    env = CellTrackingEnv(
        image_dir='synthetic_data/images',
        mask_dir='synthetic_data/masks',
        tracking_file='synthetic_data/tracking/tracking_info.txt'
    )
    policy_net = PolicyNetwork(num_cells=5)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    train(env, policy_net, optimizer, num_episodes=500)
