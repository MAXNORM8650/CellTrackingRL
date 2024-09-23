import torch
from env import CellTrackingEnv
from Policy import PolicyNetwork
import matplotlib.pyplot as plt
import cv2
import os

def test(env, policy_net):
    policy_net.eval()
    state = env.reset()
    done = False
    total_reward = 0

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(5, 5))

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = policy_net(state_tensor)
            action = mean  # Use mean as action during testing

        action_np = action.numpy()[0]
        next_state, reward, done, _ = env.step(action_np)
        total_reward += reward

        # Render the environment
        render_state(state, action_np, env.current_frame, ax)

        state = next_state

    print(f"Total Reward during testing: {total_reward:.2f}")
    plt.ioff()
    plt.show()

def render_state(state, action, frame_id, ax):
    # Create output directory if it doesn't exist
    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)

    # Draw predicted positions on the image
    image = state.copy()
    for pos in action:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    # Save the image
    cv2.imwrite(f'{output_dir}/frame_{frame_id:04d}.png', image)

if __name__ == "__main__":
    env = CellTrackingEnv(
        image_dir='synthetic_data/images',
        mask_dir='synthetic_data/masks',
        tracking_file='synthetic_data/tracking/tracking_info.txt'
    )
    policy_net = PolicyNetwork(num_cells=5)
    # Load trained weights if saved
    policy_net.load_state_dict(torch.load('policy_net.pth'))

    test(env, policy_net)
