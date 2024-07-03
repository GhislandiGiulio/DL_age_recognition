from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2

# Load the image using Pillow
image_path = 'image.png'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a NumPy array
image_np = np.array(image)

# Get the dimensions of the frame
height, width, _= image_np.shape

# Define the center and axes of the oval
center = (width // 2, height // 2)
axes = (width // 6, height // 3)  # Half the width and height of the frame

# Define the color and thickness of the oval
color = (0, 255, 0)  # Green color in BGR format
thickness = 2

cv2.ellipse(image_np, center, axes, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)


# Crop the image
# Define the crop box (left, upper, right, lower)
upper = height // 6
lower = height - (height // 6)
crop_height = lower - upper

# Calculate left and right to make the crop box square
center_x = width // 2
half_crop_width = crop_height // 2
left = max(0, center_x - half_crop_width)
right = min(width, center_x + half_crop_width)

# Define the crop box (left, upper, right, lower)
crop_box = (left, upper, right, lower)

cropped_image_np = image_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

print(cropped_image_np.shape)

# Plot the original and cropped images using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original image
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot the cropped image
axes[1].imshow(cropped_image_np)
axes[1].set_title("Cropped Image")
axes[1].axis('off')

plt.show()
