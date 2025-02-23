#!/usr/bin/env python3
import os
import imageio
import matplotlib.pyplot as plt
from scripts.preprocessing import load_and_preprocess
from scripts.segmentation import train_em_gmm, segment_image
from scripts.visualization import display_segmentation, create_gif

def main():
    # Liste des chemins d'images (chemins relatifs, style Ubuntu)
    image_paths = [
        "data/brainmri1.png",
        "data/brainmri2.png",
        "data/brainmri3.png"
    ]
    
    # Liste pour stocker les frames du GIF d'inférence
    gif_frames = []
    
    for path in image_paths:
        # Charger et prétraiter l'image avec extraction du cerveau
        image_brain, brain_mask = load_and_preprocess(path)
        
        # Entraîner le modèle EM-GMM sur l'image prétraitée
        gmm = train_em_gmm(image_brain)
        
        # Segmenter l'image en appliquant le masque
        seg_image = segment_image(gmm, image_brain, brain_mask)
        
        # Afficher la segmentation (optionnel)
        display_segmentation(image_brain, seg_image)
        
        # Générer une figure pour sauvegarder la frame du GIF
        fig, ax = plt.subplots()
        ax.imshow(seg_image, cmap='viridis')
        ax.set_title("Segmentation")
        ax.axis("off")
        temp_path = "temp_frame.png"
        plt.savefig(temp_path)
        plt.close(fig)
        gif_frames.append(imageio.imread(temp_path))
        os.remove(temp_path)
    
    # Créer le dossier artifacts s'il n'existe pas
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    
    # Créer le GIF d'inférence et le sauvegarder dans artifacts
    create_gif(gif_frames, "artifacts/gmm_inference.gif", duration=0.5)
    print("GIF d'inférence sauvegardé dans 'artifacts/gmm_inference.gif'.")

if __name__ == "__main__":
    main()
