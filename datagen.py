import cv2
import numpy as np
import os

def generate_synthetic_dataset(output_dir, num_frames=50, image_size=(128, 128), num_cells=5):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/tracking", exist_ok=True)

    cell_positions = [np.array([np.random.randint(20, image_size[0]-20),
                                np.random.randint(20, image_size[1]-20)]) for _ in range(num_cells)]
    cell_velocities = [np.random.randint(-2, 3, size=2) for _ in range(num_cells)]
    tracking_info = {}

    for frame in range(num_frames):
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

        for cell_id, (pos, vel) in enumerate(zip(cell_positions, cell_velocities)):
            # Update cell position
            pos += vel

            # Ensure cells stay within bounds
            pos = np.clip(pos, 10, image_size[0]-10)

            # Draw the cell on the image
            cv2.circle(image, tuple(pos), 10, (255, 255, 255), -1)
            cv2.circle(mask, tuple(pos), 10, cell_id+1, -1)  # Cell IDs start from 1

            # Record tracking info
            if cell_id not in tracking_info:
                tracking_info[cell_id] = []
            tracking_info[cell_id].append((frame, pos[0], pos[1]))

        # Save the image and mask
        cv2.imwrite(f"{output_dir}/images/frame_{frame:04d}.png", image)
        cv2.imwrite(f"{output_dir}/masks/mask_{frame:04d}.png", mask)

    # Save tracking info
    with open(f"{output_dir}/tracking/tracking_info.txt", 'w') as f:
        for cell_id, positions in tracking_info.items():
            for pos in positions:
                frame_id, x, y = pos
                f.write(f"{cell_id},{frame_id},{x},{y}\n")

# Generate the dataset
generate_synthetic_dataset('synthetic_data')
