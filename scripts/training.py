# from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import cv2

def train_em_gmm(images, n_components=3, max_iter=100):
    gif_frames = []
    seg_images = []  # Store segmented images
    gmm_model = None  # To store the GMM model
    
    cluster_histograms = []  # To store histograms of clusters
    log_likelihoods = []  # To store log-likelihood values per iteration
    
    for img in images:
        # Flatten image and apply reshaping
        img_flat = img.flatten()
        img_reduced = img_flat.reshape(-1, 1)  # Reshape for compatibility with GMM

        # Initialize GMM
        gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
        
        # Track log-likelihood and cluster distributions during training
        for iteration in range(max_iter):
            gmm.fit(img_reduced)
            
            # Track log-likelihood values during training
            log_likelihoods.append(gmm.lower_bound_)
            
            # Save the cluster histograms (intensity distribution)
            histograms = [np.histogram(gmm.sample(1000)[0], bins=50, range=(0, 255))[0] for _ in range(n_components)]
            cluster_histograms.append(histograms)
        
        # Final GMM model
        gmm_model = gmm

        # Predict segmentation and reshape back to original image dimensions
        seg_result = gmm.predict(img_reduced)
        seg_image = seg_result.reshape(img.shape)

        seg_images.append(seg_image)

        # Plot for GIF
        fig, ax = plt.subplots()
        ax.imshow(seg_image, cmap='viridis')
        ax.set_title(f'Segmentation')
        plt.axis('off')

        frame_path = f"temp_frame_{len(gif_frames)}.png"
        plt.savefig(frame_path)
        gif_frames.append(imageio.v2.imread(frame_path))
        plt.close(fig)
        os.remove(frame_path)

    return gmm_model, gif_frames, cluster_histograms, log_likelihoods





# Train the EM-GMM and generate frames for GIF

# Load and resize images
# desired_resolution = (179, 214)  # Use a common resolution for all images
top_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri1.png', cv2.IMREAD_GRAYSCALE)
sagittal_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri2.png', cv2.IMREAD_GRAYSCALE)
rear_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri3.png', cv2.IMREAD_GRAYSCALE)

images = [top_view_norm, sagittal_view_norm, rear_view_norm]
gmm_model, gif_frames, cluster_histograms, log_likelihoods = train_em_gmm(images)

# Save GIF
output_gif = 'C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\artifacts\\gmm_training_progress.gif'
imageio.v2.mimsave(output_gif, gif_frames, duration=0.5)
