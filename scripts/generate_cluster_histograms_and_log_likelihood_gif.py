import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from sklearn.mixture import GaussianMixture
import cv2
from training import train_em_gmm  # Assuming this imports your EM-GMM training function

def generate_histogram_and_ll_gif(images, gmm_model, output_gif="histograms_and_ll.gif"):
    gif_frames = []
    log_likelihoods = []  # To track log-likelihood during iterations
    n_components = gmm_model.n_components  # Number of components/clusters
    
    # Prepare the output directory for frames
    if not os.path.exists("temp_frames"):
        os.makedirs("temp_frames")
    
    for img in images:
        # Flatten the image for prediction
        img_flat = img.flatten().reshape(-1, 1)
        
        # Plot histograms for each component
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Track the Log-Likelihood for each iteration
        log_likelihood = []
        
        # For histogram, we'll extract the data points corresponding to each cluster
        responsibilities = gmm_model.predict_proba(img_flat)
        for i in range(n_components):
            cluster_data = img_flat[responsibilities[:, i] > 0.5]  # Filter by responsibility threshold
            ax1.hist(cluster_data, bins=30, alpha=0.5, label=f'Cluster {i+1}')
        
        ax1.set_title("Cluster Histograms")
        ax1.legend()

        # Log-Likelihood graph
        ax2.plot(log_likelihoods, color='blue')
        ax2.set_title('Log-Likelihood vs. Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Log-Likelihood')
        
        # Store the Log-Likelihood for the current image
        log_likelihood.append(gmm_model.lower_bound_)
        
        # Save the current frame
        frame_path = f"temp_frames/frame_{len(gif_frames)}.png"
        plt.savefig(frame_path)
        gif_frames.append(imageio.v2.imread(frame_path))
        plt.close(fig)
    
    # Create and save the GIF
    imageio.v2.mimsave(output_gif, gif_frames, duration=0.5)
    
    # Clean up frames
    for frame in os.listdir("temp_frames"):
        os.remove(os.path.join("temp_frames", frame))
    os.rmdir("temp_frames")

# Load and preprocess images
top_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri1.png', cv2.IMREAD_GRAYSCALE)
sagittal_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri2.png', cv2.IMREAD_GRAYSCALE)
rear_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri3.png', cv2.IMREAD_GRAYSCALE)

images = [top_view_norm, sagittal_view_norm, rear_view_norm]

# Train the GMM model
gmm_model, _ = train_em_gmm(images)

# Run the function to generate the histograms and log-likelihood GIF
generate_histogram_and_ll_gif(images, gmm_model)
