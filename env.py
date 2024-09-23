import numpy as np
import cv2
import os
import gymnasium as gym
from gymnasium import spaces
class CellTrackingEnv(gym.Env):
    def __init__(self, image_dir, mask_dir, tracking_file):
        super(CellTrackingEnv, self).__init__()
        
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.png') and fname.startswith('frame')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, fname)
            for fname in os.listdir(mask_dir)
            if fname.endswith('.png') and fname.startswith('mask')
        ])
        self.num_frames = len(self.image_paths)
        
        # Load tracking information
        self.tracking_data = self.load_tracking_info(tracking_file)
        
        # Define the state and action spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=0, high=127, shape=(5, 2), dtype=np.float32  # Predict positions for 5 cells
        )
        
        # Initialize state
        self.current_frame = 0

    def load_tracking_info(self, tracking_file):
        tracking_data = {}
        with open(tracking_file, 'r') as f:
            for line in f:
                cell_id, frame_id, x, y = map(int, line.strip().split(','))
                if cell_id not in tracking_data:
                    tracking_data[cell_id] = {}
                tracking_data[cell_id][frame_id] = (x, y)
        return tracking_data

    def reset(self):
        self.current_frame = 0
        state = cv2.imread(self.image_paths[self.current_frame], cv2.IMREAD_COLOR)
        if state is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[self.current_frame]}")
        return state

    def step(self, action):
        # Move to the next frame
        self.current_frame += 1
        done = self.current_frame >= self.num_frames - 1
        if not done:
            state = cv2.imread(self.image_paths[self.current_frame], cv2.IMREAD_COLOR)
            if state is None:
                raise FileNotFoundError(f"Image not found: {self.image_paths[self.current_frame]}")
        else:
            state = None

        # Calculate reward
        reward = self.calculate_reward(action)
        return state, reward, done, {}

    def calculate_reward(self, action):
        # Calculate negative distance between predicted and true positions
        reward = 0.0
        for cell_id in range(len(action)):
            if cell_id in self.tracking_data:
                true_pos = self.tracking_data[cell_id].get(self.current_frame)
                if true_pos:
                    pred_pos = action[cell_id]
                    dist = np.linalg.norm(np.array(true_pos) - pred_pos)
                    reward -= dist  # Negative distance as penalty
        return reward

    def render(self, mode='human'):
        image = cv2.imread(self.image_paths[self.current_frame], cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imshow('Cell Tracking', image)
            cv2.waitKey(1)
