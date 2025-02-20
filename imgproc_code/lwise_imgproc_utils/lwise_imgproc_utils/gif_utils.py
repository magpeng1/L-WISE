import numpy as np


# Function to add a border to an image
def add_border(img_array, color, border_width=2):
    h, w, _ = img_array.shape
    new_img = np.zeros((h + 2*border_width, w + 2*border_width, 3), dtype=np.uint8)
    new_img[border_width:-border_width, border_width:-border_width] = img_array
    new_img[:border_width, :, :] = color  # Top border
    new_img[-border_width:, :, :] = color  # Bottom border
    new_img[:, :border_width, :] = color  # Left border
    new_img[:, -border_width:, :] = color  # Right border
    return new_img


# Function to compute the border color based on the frame index
def compute_border_color(frame_idx, num_frames):
    # Transition from red to bright green over the original sequence
    fraction = frame_idx / (num_frames - 1)  # Use num_frames - 1 to ensure the last frame gets full green
    color = np.array([255 * (1 - fraction), 255 * fraction, 0], dtype=np.uint8)
    return color
