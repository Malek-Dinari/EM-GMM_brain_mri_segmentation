import imageio
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Import the trained GMM model
from training import gmm_model

def generate_inference_gif(model, images, output_gif="gmm_inference.gif"):
    gif_frames = []
    cluster_histograms = []  # Store the histograms during inference
    log_likelihoods = []  # Store log-likelihoods during inference
    
    # Track the log-likelihood for each iteration in the EM algorithm
    for img in images:
        # Flatten the image and predict the segmentations
        seg_result = model.predict(img.flatten().reshape(-1, 1))  # Flatten and reshape to match GMM input shape
        seg_image = seg_result.reshape(img.shape)  # Reshape back to the original image shape
        
        # Update the log-likelihood and cluster histograms
        log_likelihoods.append(model.lower_bound_)  # Track the log-likelihood value after EM step
        histograms = [np.histogram(seg_image.flatten(), bins=50, range=(0, 255))[0] for _ in range(3)]
        cluster_histograms.append(histograms)
        
        # Visualize the segmentation
        fig, ax = plt.subplots()
        ax.imshow(seg_image, cmap='cividis')  # Change colormap to 'cividis'
        ax.set_title('Inference')
        plt.axis('off')
        
        # Save frame for gif
        frame_path = "temp_inference_frame.png"
        plt.savefig(frame_path)
        gif_frames.append(imageio.v2.imread(frame_path))
        plt.close(fig)
        os.remove(frame_path)
    
    # Plotting the results with subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 14))  # Adjusted figure size for better spacing
    
    # First row: Segmented images
    for i, ax in enumerate(axs[0, :3]):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Original Image {i+1}')
        plt.axis('off')
    
    for i, ax in enumerate(axs[1, :3]):
        ax.imshow(gif_frames[i], cmap='cividis')
        ax.set_title(f'Segmented Image {i+1}')
        plt.axis('off')

    # Second row: Histograms and Log-Likelihood
    for i, hist in enumerate(cluster_histograms):
        axs[2, 0].hist(hist[0], bins=50, alpha=0.7, label=f"Cluster {i+1}")
    axs[2, 0].set_title('Cluster Intensity Distributions')
    axs[2, 0].legend(loc='best')
    
    # Improved Log-Likelihood Plot
    axs[2, 1].plot(log_likelihoods, marker='o', color='b', linestyle='-', markersize=5)  # Add marker for LL values
    axs[2, 1].set_title('Log-Likelihood vs Iterations')
    axs[2, 1].set_xlabel('Iteration')
    axs[2, 1].set_ylabel('Log-Likelihood')
    
    # Save and show the final result
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    plt.savefig('final_result.png')
    plt.show()

    # Save the final gif
    imageio.v2.mimsave(output_gif, gif_frames, duration=0.5)

# Load and preprocess images
top_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri1.png', cv2.IMREAD_GRAYSCALE)
sagittal_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri2.png', cv2.IMREAD_GRAYSCALE)
rear_view_norm = cv2.imread('C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\EM-GMM-segment-mri\\data\\brainmri3.png', cv2.IMREAD_GRAYSCALE)

images = [top_view_norm, sagittal_view_norm, rear_view_norm]

# Run inference and create the GIF
generate_inference_gif(gmm_model, images)
