import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
top_view = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri1.png', cv2.IMREAD_GRAYSCALE)
sagittal_view = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri2.png', cv2.IMREAD_GRAYSCALE)
rear_view = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri3.png', cv2.IMREAD_GRAYSCALE)

# Resize images to a common resolution (e.g., 200x200)
desired_resolution = (200, 200)

top_view_resized = cv2.resize(top_view, desired_resolution)
sagittal_view_resized = cv2.resize(sagittal_view, desired_resolution)
rear_view_resized = cv2.resize(rear_view, desired_resolution)

# Normalize the images
top_view_norm = top_view_resized / 255.0
sagittal_view_norm = sagittal_view_resized / 255.0
rear_view_norm = rear_view_resized / 255.0

# Show the images
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes[0].imshow(top_view_norm, cmap='gray')
axes[0].set_title('Top View')
axes[1].imshow(sagittal_view_norm, cmap='gray')
axes[1].set_title('sagittal View')
axes[2].imshow(rear_view_norm, cmap='gray')
axes[2].set_title('Rear View')

plt.show()
